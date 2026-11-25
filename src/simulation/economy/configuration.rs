use super::transport::railway;
use crate::simulation::{
    Device, negotiation::BargainingConfig, storage::layout::WORD_BITS, system::AsBufferArg,
};
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
    pub baking_minutes: u32,
    pub cultivation_minutes: u32,
    pub harvest_start: u32,
    pub harvest_days: u32,
    pub annual_yield_per_plot: u32,
    pub grain_price: u32,
}
impl HomeProduction {
    fn new(value: crate::domain::enterprise::HouseholdProduction, grain_price: u32) -> Self {
        match value {
            crate::domain::enterprise::HouseholdProduction::Unavailable => Self {
                harvest_start: crate::domain::calendar::HARVEST_START,
                harvest_days: crate::domain::calendar::HARVEST_DAYS,
                ..Self::default()
            },
            crate::domain::enterprise::HouseholdProduction::Available {
                baking_minutes,
                cultivation_minutes,
                harvest_start,
                harvest_days,
                annual_yield_per_plot,
                ..
            } => Self {
                baking_minutes,
                cultivation_minutes,
                harvest_start,
                harvest_days,
                annual_yield_per_plot,
                grain_price,
            },
        }
    }
    pub fn launch(self) -> HomeProductionLaunch {
        HomeProductionLaunch {
            baking_minutes: self.baking_minutes,
            cultivation_minutes: self.cultivation_minutes,
            harvest_start: self.harvest_start,
            harvest_days: self.harvest_days,
            annual_yield_per_plot: self.annual_yield_per_plot,
            grain_price: self.grain_price,
        }
    }
}
#[derive(CubeType, CubeLaunch)]
pub struct EconomyConfig {
    pub heating_per_person: u32,
    pub actors: u32,
    pub state_stride: u32,
    pub fields: u32,
    pub ring: u32,
    pub grain_rounds: u32,
    pub first_firm: u32,
    pub retail_rotation: u32,
    pub search_radius: u32,
    pub map_size: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
    pub railway: railway::RailNetwork,
    pub household_need: u32,
    pub food_medium_salary: u32,
    pub food_rich_salary: u32,
    #[cube(comptime)]
    pub food_income: crate::domain::food::FoodIncome,
    pub wheat_kg_per_hectare: u32,
    #[cube(comptime)]
    pub commercial_wheat: crate::domain::food::WheatHusbandry,
    #[cube(comptime)]
    pub economy: crate::domain::Economy,
    #[cube(comptime)]
    pub cohorts: Feature,
    pub home: HomeProduction,
    pub opening_prices: Box<[u32]>,
    pub market_table: Box<[u32]>,
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
struct CartageAccounts {
    accounts: burn_wgpu::CubeTensor,
    firms: u32,
}

impl CartageAccounts {
    fn new(device: &Device, state: &State) -> Self {
        let mut carriers = state
            .actors
            .firm_indices()
            .filter(|&row| state.actors.at(row).kind == Good::Railway as u32)
            .map(|row| row as u32)
            .collect::<Vec<_>>();
        let cartage_firms = carriers.len() as u32;
        carriers.extend(
            state
                .actors
                .iter()
                .enumerate()
                .filter(|(_, actor)| actor.kind == 0)
                .map(|(row, _)| row as u32),
        );
        Self {
            accounts: device.tensor(&carriers),
            firms: cartage_firms,
        }
    }
}

#[cube]
impl EconomyConfig {
    pub fn query_members(&self) -> u32 {
        let cells = self.spatial_side * self.spatial_side;
        3 * (cells + 1 + self.actors)
            + crate::simulation::storage::words::ceildiv(
                self.actors,
                crate::simulation::storage::layout::DISCOVERY_BLOCK,
            )
            + 1
    }

    pub fn coordinate(&self, field: u32, actor: u32) -> u32 {
        field * self.state_stride + actor
    }
}

pub struct Configuration {
    pub heating_per_person: u32,
    rentals: crate::domain::landlords::RentalRegistry,
    pub actors: u32,
    pub state_stride: u32,
    pub fields: u32,
    pub ring: u32,
    pub grain_rounds: u32,
    pub first_firm: u32,
    pub retail_rotation: u32,
    pub search_radius: u32,
    pub map_size: u32,
    pub commute_speed: u32,
    pub shipment_speed: u32,
    pub railway: railway::Network,
    pub household_need: u32,
    pub food_medium_salary: u32,
    pub food_rich_salary: u32,
    pub food_income: crate::domain::food::FoodIncome,
    pub wheat_kg_per_hectare: u32,
    pub commercial_wheat: crate::domain::food::WheatHusbandry,
    pub economy: crate::domain::Economy,
    pub cohorts: Feature,
    pub home: HomeProduction,
    opening_prices: burn_wgpu::CubeTensor,
    pub market_table: burn_wgpu::CubeTensor,
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
    pub farm_market: super::market::farm_offers::FarmMarket,
    pub cartage_accounts: burn_wgpu::CubeTensor,
    pub cartage_firms: u32,
    pub rental_accounts: burn_wgpu::CubeTensor,
    pub rental_businesses: u32,
    pub small_business_expansion: u32,
    pub industrial_expansion: u32,
}
impl Configuration {
    pub(in crate::simulation) fn graph_buffers(
        &self,
    ) -> impl Iterator<Item = &burn_wgpu::CubeTensor> + Clone {
        [
            &self.opening_prices,
            &self.market_table,
            &self.price_table,
            &self.property_regions,
            &self.grain_farms,
            &self.rental_accounts,
            &self.cartage_accounts,
        ]
        .into_iter()
        .chain(self.farm_market.graph_buffers())
        .chain(self.railway.graph_buffers())
    }

    pub fn new(device: &Device, scenario: &Scenario, state: &State) -> Self {
        let rentals = crate::domain::landlords::RentalRegistry::new(scenario, state);
        let railway = railway::Network::new(device, scenario, state);
        Self::with_rentals(
            device,
            scenario,
            state,
            rentals,
            railway,
            CartageAccounts::new(device, state),
        )
    }

    #[cfg(test)]
    pub fn updated(
        &self,
        device: &Device,
        scenario: &Scenario,
        previous: &crate::domain::ActorRows,
        state: &State,
    ) -> Self {
        self.updated_using(
            device,
            scenario,
            previous,
            state,
            crate::domain::landlords::RentalChanges::Compared,
        )
    }

    pub fn updated_for_registry(
        &self,
        device: &Device,
        scenario: &Scenario,
        previous: &crate::domain::ActorRows,
        state: &State,
        changes: &[usize],
    ) -> Self {
        self.updated_using(
            device,
            scenario,
            previous,
            state,
            crate::domain::landlords::RentalChanges::Selected(changes),
        )
    }

    fn updated_using(
        &self,
        device: &Device,
        scenario: &Scenario,
        previous: &crate::domain::ActorRows,
        state: &State,
        changes: crate::domain::landlords::RentalChanges<'_>,
    ) -> Self {
        let profile = std::env::var_os("BAD_ECON_PROFILE_STAGES").is_some();
        let started = std::time::Instant::now();
        let cartage = match changes {
            crate::domain::landlords::RentalChanges::Selected(_)
                if state
                    .actors
                    .iter_from(previous.len())
                    .all(|actor| actor.kind > 0 && actor.kind != Good::Railway as u32) =>
            {
                CartageAccounts {
                    accounts: self.cartage_accounts.clone(),
                    firms: self.cartage_firms,
                }
            }
            _ => CartageAccounts::new(device, state),
        };
        let rentals = self.rentals.updated(scenario, previous, state, changes);
        if profile {
            eprintln!(
                "CONFIG_PART Rentals {:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let started = std::time::Instant::now();
        let railway = self.railway.updated(device, scenario, state);
        if profile {
            eprintln!(
                "CONFIG_PART Railway {:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let started = std::time::Instant::now();
        let updated = Self::with_rentals(device, scenario, state, rentals, railway, cartage);
        if profile {
            eprintln!(
                "CONFIG_PART Buffers {:.3}",
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        updated
    }

    pub fn update_rentals(
        &mut self,
        device: &Device,
        scenario: &Scenario,
        state: &State,
        changes: impl IntoIterator<Item = usize>,
    ) {
        match self.rentals.refresh(scenario, state, changes) {
            crate::domain::landlords::RentalRefresh::Current => (),
            crate::domain::landlords::RentalRefresh::Changed => {
                let words = self.rentals.words();
                let businesses = self.rentals.accounts.len() as u32;
                match (
                    self.rental_accounts.meta.num_elements(),
                    self.rental_businesses,
                ) == (words.len(), businesses)
                {
                    true => device.write(&self.rental_accounts, &words),
                    false => self.rental_accounts = device.tensor(&words),
                }
                self.rental_businesses = businesses;
            }
        }
    }

    fn with_rentals(
        device: &Device,
        scenario: &Scenario,
        state: &State,
        rentals: crate::domain::landlords::RentalRegistry,
        railway: railway::Network,
        cartage: CartageAccounts,
    ) -> Self {
        let config = scenario.config();
        let collection_days = config
            .distributors
            .iter()
            .filter(|store| store.good == Good::Grain)
            .map(|store| store.reserve_days.min(config.coverage.max(1)) as usize)
            .sum::<usize>();
        let grain_rounds = match (config.economy, collection_days) {
            (crate::domain::Economy::Industrial, 1..) => {
                scenario.population_rows().div_ceil(collection_days).clamp(
                    super::policy::MATCHING_ROUNDS,
                    super::policy::MAX_GRAIN_COLLECTION_ROUNDS,
                )
            }
            _ => super::policy::MATCHING_ROUNDS,
        };
        Self {
            heating_per_person: crate::domain::calendar::Weather::on_day(state.day)
                .heating_units_per_person,
            small_business_expansion: config.expansion.small_business_per_mille,
            industrial_expansion: config.expansion.industrial_per_mille,
            cartage_accounts: cartage.accounts,
            cartage_firms: cartage.firms,
            rental_accounts: device.tensor(&rentals.words()),
            rental_businesses: rentals.accounts.len() as u32,
            rentals,
            actors: state.actors.len() as u32,
            state_stride: state.actors.len() as u32,
            fields: scenario.fields() as u32,
            ring: scenario.ring() as u32,
            grain_rounds: grain_rounds as u32,
            first_firm: scenario.population_rows() as u32,
            retail_rotation: (config.distributors.len() as u32)
                .div_ceil(
                    super::policy::DISCOVERY_CANDIDATES.saturating_mul(
                        config
                            .map_size
                            .div_ceil(config.search_radius)
                            .saturating_pow(2),
                    ),
                )
                .clamp(1, Good::COUNT as u32),
            search_radius: config.search_radius,
            map_size: config.map_size,
            commute_speed: config.commute_speed,
            shipment_speed: config.shipment_speed,
            railway,
            household_need: config.household_need,
            food_medium_salary: config.food.map_or(0, |food| food.medium_daily_salary),
            food_rich_salary: config.food.map_or(0, |food| food.rich_daily_salary),
            food_income: config
                .food
                .map_or(crate::domain::food::FoodIncome::Nominal, |food| food.income),
            wheat_kg_per_hectare: config.food.map_or(0, |food| food.wheat_kg_per_hectare),
            commercial_wheat: config
                .food
                .map(|food| food.commercial_wheat)
                .unwrap_or_default(),
            economy: config.economy,
            cohorts: Feature::from(config.people.as_ref()),
            home: HomeProduction::new(config.household_production, config.prices[0]),
            property_mode: Feature::from(config.property.as_ref()),
            farm_market: super::market::farm_offers::FarmMarket::new(
                device,
                &state.actors,
                config.map_size,
            ),
            grain_farms: device.tensor(
                &state
                    .actors
                    .firm_indices()
                    .filter(|&index| state.actors.at(index).kind == Good::Grain as u32)
                    .map(|index| index as u32)
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
                                    .index_of(region.landowner)
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
            opening_prices: device.tensor(&Good::ALL.map(|good| scenario.initial_price(good))),
            market_table: device.tensor(
                &Good::ALL
                    .map(Good::consumption_interval)
                    .into_iter()
                    .chain(std::iter::repeat_n(0, 5 + 2 * state.actors.len()))
                    .collect::<Vec<_>>(),
            ),
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
            rental_accounts: self.rental_accounts.buffer_arg(),
            rental_businesses: self.rental_businesses,
            property_mode: self.property_mode,
            property_regions: self.property_regions.buffer_arg(),
            grain_farms: self.grain_farms.buffer_arg(),
            actors: self.actors,
            state_stride: self.state_stride,
            fields: self.fields,
            ring: self.ring,
            grain_rounds: self.grain_rounds,
            first_firm: self.first_firm,
            retail_rotation: self.retail_rotation,
            search_radius: self.search_radius,
            map_size: self.map_size,
            commute_speed: self.commute_speed,
            shipment_speed: self.shipment_speed,
            railway: self.railway.launch(),
            household_need: self.household_need,
            food_medium_salary: self.food_medium_salary,
            food_rich_salary: self.food_rich_salary,
            food_income: self.food_income,
            wheat_kg_per_hectare: self.wheat_kg_per_hectare,
            commercial_wheat: self.commercial_wheat,
            economy: self.economy,
            cohorts: self.cohorts,
            home: self.home.launch(),
            opening_prices: self.opening_prices.buffer_arg(),
            market_table: self.market_table.buffer_arg(),
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
            price_table: self.price_table.buffer_arg(),
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
            let previous_accounts = cached.rental_accounts.clone();
            let previous_businesses = cached.rental_businesses;
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
            if (
                previous.len(),
                previous_accounts.meta.num_elements(),
                previous_businesses,
            ) == (
                state.actors.len(),
                rebuilt.rental_accounts.meta.num_elements(),
                rebuilt.rental_businesses,
            ) {
                assert!(crate::simulation::gpu::tensor::same_buffer(
                    &previous_accounts,
                    &cached.rental_accounts,
                ));
            }
            assert_eq!(cached.rental_businesses, rebuilt.rental_businesses);
            assert_eq!(
                device.read(&cached.rental_accounts)?,
                device.read(&rebuilt.rental_accounts)?
            );
        }
        Ok(())
    }
}

use super::{Experiment, Scenario, ScenarioConfig};
use crate::{
    arithmetic::add,
    domain::{
        CohortSizes, Economy, Field, Good, HouseholdLiquidity, ProductionMethod, RentPolicy, State,
        Technology, employment, enterprise, food, prices, railway, research, supply,
    },
};
use anyhow::{Result, anyhow};

impl Scenario {
    pub fn industrial(households: u32, seed: u64, experiment: Experiment) -> Result<Self> {
        Self::industrial_population(households, seed, experiment, None)
    }

    pub fn for_people(
        people: u32,
        sizes: CohortSizes,
        economy: Economy,
        seed: u64,
        experiment: Experiment,
    ) -> Result<Self> {
        match economy {
            Economy::Industrial => {
                Self::industrial_population(people, seed, experiment, Some(sizes))
            }
            Economy::Agrarian => {
                let mut config = ScenarioConfig::baseline(people, seed, experiment)?;
                config.people = Some(people);
                config.cohort_sizes = sizes;
                Self::new(config)
            }
        }
    }

    fn industrial_population(
        households: u32,
        seed: u64,
        experiment: Experiment,
        cohorts: Option<CohortSizes>,
    ) -> Result<Self> {
        if cohorts.is_none() && !ScenarioConfig::HOUSEHOLDS.contains(&households) {
            Err(anyhow!("legacy household count must be in 1..=1000000"))?;
        }
        let mut config = ScenarioConfig::baseline(households, seed, experiment)?;
        config.people = cohorts.map(|_| households);
        config.cohort_sizes = cohorts.unwrap_or_default();
        config.economy = Economy::Industrial;
        config.rent = Some(RentPolicy::default());
        let transport_scale = config.map_size / 1000;
        config.search_radius *= transport_scale;
        config.commute_speed *= transport_scale;
        config.shipment_speed = match experiment {
            Experiment::Transport => 50,
            _ => 100,
        } * transport_scale;
        config.demography.initial_household_members = if cohorts.is_some() { 1 } else { 3 };
        config.household_production = enterprise::HouseholdProduction::industrial();
        config.firms = config.firms.max(3 * Good::COUNT as u32);
        config.capacity = 1000u32.max(households.div_ceil(100_000)).min(10_000);
        config.firm_cash = 1_000_000;
        config.reserve = config.firm_cash;
        config.coverage = 6;
        config.producer_buffer_days = Some(2);
        config.buffer_days = match experiment {
            Experiment::Shortage => 0,
            _ => 6,
        };
        config.packet_minutes = 20;
        config.packet_output = 20;
        config.durable_packet_output = Some(1);
        config.wage = 12;
        config.household_cash = 7
            * (u64::from(config.household_need)
                * u64::from(config.demography.initial_household_members)
                * u64::from(food::FoodPolicy::PRICES[2])
                + u64::from(config.rent.map_or(0, |rent| rent.household_daily)));
        config.food = Some(food::FoodPolicy::industrial());
        if let enterprise::HouseholdProduction::Available {
            annual_yield_per_plot,
            ..
        } = &mut config.household_production
        {
            *annual_yield_per_plot = food::FoodPolicy::default().wheat_kg_per_hectare;
        }
        config.prices = prices::StartingPrices::production(&config).food();
        config.household_liquidity = HouseholdLiquidity::PaycheckToPaycheck;
        config.behavior.wage_increment = 0;
        config.behavior.price_response = 1000;
        config.distributors = (0..config.firms)
            .map(|firm| (firm, generated_good(&config, firm as usize)))
            .filter(|&(firm, good)| match good {
                Good::Bread => firm % 3 == 0,
                Good::Grain => firm % 3 == 1,
                _ => false,
            })
            .map(|(firm, good)| enterprise::Distributor {
                firm: u64::from(households + firm + 1),
                good,
                handling_minutes: 5,
                units_per_shift: 24,
                reserve_days: 3,
                margin_per_mille: 250,
            })
            .collect();
        let labs = (0..config.firms.min(households))
            .filter(|&firm| {
                !matches!(
                    generated_good(&config, firm as usize),
                    Good::Housing | Good::Railway
                ) && !config
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
                        minimum_effort: 30_000,
                        pioneer_effort: 60_000,
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
        let mut scenario = Self::new(config)?;
        let railway =
            railway::RailwayConfig::generated(&scenario.initial_actors, scenario.config.map_size);
        scenario.set_railway(railway)?;
        Ok(scenario)
    }

    pub(super) fn generated(&self) -> Result<State> {
        let actors = self.actors();
        let config = &self.config;
        let food_reserve = supply::StartingReserves::new(config);
        let settlement = employment::InitialSettlement::new(config);
        let mut state = self.empty_opening_state(self.next_identity());
        (0..actors).try_for_each(|actor| -> Result<()> {
            let entry = state.actors.at(actor);
            let kind = entry.kind;
            let retailer = self.distributor(entry.identity);
            let cash = match kind {
                0 => match &self.cohorts {
                    Some(layout) => layout.rows[actor].cash,
                    None => self.initial_employment.cash(config, actor)?,
                },
                _ => config.firm_cash,
            };
            state.set(Field::Cash, actor, cash);
            let farmers = match (kind, &self.cohorts) {
                (0, Some(layout)) => Some(layout.rows[actor].farmers),
                (0, None) => Some(u32::from(settlement.worker(actor).is_none())),
                _ => None,
            };
            if let Some(farm) =
                farmers.and_then(|farmers| supply::StartingFarm::new(config, farmers))
            {
                state.set(Field::LandPlots, actor, farm.plots);
                for stock in farm.inventory {
                    stock.seed(&mut state, actor)?;
                }
            }
            state.set(
                Field::EnterpriseStatus,
                actor,
                enterprise::EnterpriseState::Operating as u64,
            );
            state.set(Field::Wage, actor, u64::from(config.wage));
            match kind {
                0 => {
                    state.set(Field::Price, actor, config.prices[2].into());
                    for stock in food_reserve.household(u64::from(entry.capacity)) {
                        stock.seed(&mut state, actor)?;
                    }
                }
                _ => {
                    let output = Good::from_index(kind)?;
                    for stock in self
                        .production
                        .inventory(config, &food_reserve, output, retailer)
                    {
                        stock.seed(&mut state, actor)?;
                    }
                    state.set(
                        Field::Price,
                        actor,
                        u64::from(match retailer {
                            Some(store) => self.starting_prices.retail(store, config),
                            None => self.initial_price(output),
                        }),
                    );
                    state.set(
                        Field::Demand,
                        actor,
                        match config.economy {
                            Economy::Agrarian => {
                                u64::from(entry.capacity) * u64::from(config.packet_output)
                            }
                            Economy::Industrial => self.production.daily[kind as usize - 1],
                        },
                    );
                }
            }
            state.initial_cash = add(state.initial_cash, cash)?;
            Ok(())
        })?;
        Ok(state)
    }
}

pub(in crate::domain) fn generated_firm_identity(config: &ScenarioConfig, firm: usize) -> u64 {
    u64::from(config.population_size()) + firm as u64 + 1
}

pub(in crate::domain) fn generated_good(config: &ScenarioConfig, firm: usize) -> Good {
    match config.economy {
        Economy::Agrarian => Good::ALL[firm % 3],
        Economy::Industrial => match firm {
            index if index < Good::COUNT => Good::ALL[index],
            _ => match (firm - Good::COUNT) % 20 {
                0..=3 => Good::Grain,
                4..=7 => Good::Flour,
                8..=11 => Good::Bread,
                slot => {
                    Good::ALL[3 + ((firm - Good::COUNT) / 20 * 8 + slot - 12) % (Good::COUNT - 3)]
                }
            },
        },
    }
}

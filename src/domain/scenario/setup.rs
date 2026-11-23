use super::{Scenario, ScenarioConfig, generated_firm_identity, generated_good};
use crate::{
    arithmetic::add,
    domain::{
        Good, InitialTotals, cohorts, employment, enterprise, finance, prices, railway, research,
        resources, supply,
    },
};
use anyhow::{Context, Result, anyhow};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct GrainMarkets {
    pub sites: BTreeMap<usize, usize>,
    pub villages: Vec<usize>,
}

impl GrainMarkets {
    fn new(config: &ScenarioConfig, retailers: &BTreeMap<u64, enterprise::Distributor>) -> Self {
        match (config.version, config.economy, &config.population) {
            (5.., crate::domain::Economy::Industrial, None) => {
                let rows = |good| {
                    (0..config.firms as usize)
                        .filter(move |&firm| generated_good(config, firm) == good)
                        .map(move |firm| config.population_size() as usize + firm)
                };
                let dealer = |row: &usize| retailers.get(&(*row as u64 + 1));
                let villages = rows(Good::Grain)
                    .filter(|row| dealer(row).is_some_and(|store| store.good == Good::Grain))
                    .collect();
                let sites = rows(Good::Flour)
                    .zip(rows(Good::Grain))
                    .filter(|(mill, _)| dealer(mill).is_none())
                    .chain(rows(Good::Bread).zip(rows(Good::Grain)))
                    .collect();
                Self { sites, villages }
            }
            _ => Self::default(),
        }
    }
}

impl Scenario {
    pub fn new(config: ScenarioConfig) -> Result<Self> {
        let config = config.normalized()?;
        let retailers = config
            .distributors
            .iter()
            .cloned()
            .map(|store| (store.firm, store))
            .collect::<BTreeMap<_, _>>();
        match retailers.len() == config.distributors.len() {
            true => (),
            false => Err(anyhow!("distributor identities must be unique"))?,
        }
        let production = supply::StartingProduction::new(&config, &retailers);
        let initial_employment =
            employment::InitialEmployment::new(&config, &production, &retailers);
        let totals = match &config.population {
            Some(population) => population.totals()?,
            None => generated_totals(&config, &production, &initial_employment, &retailers)?,
        };
        let resources = resources::ResourceGeography::new(&config)?;
        let grain_markets = GrainMarkets::new(&config, &retailers);
        let mut scenario = Self {
            starting_prices: prices::StartingPrices::configured(&config),
            config,
            totals,
            resources,
            rail_geography: railway::RailGeography::default(),
            production,
            retailers,
            grain_markets,
            initial_actors: std::sync::Arc::default(),
            initial_employment,
            cohorts: None,
        };
        scenario.cohorts = scenario
            .config
            .people
            .map(|_| cohorts::CohortLayout::new(&scenario).map(std::sync::Arc::new))
            .transpose()?;
        if let (Some(layout), None) = (&scenario.cohorts, &scenario.config.population) {
            scenario.totals.cash = layout
                .rows
                .iter()
                .try_fold(scenario.totals.cash, |total, row| add(total, row.cash))?;
        }
        scenario.initial_actors = std::sync::Arc::new(
            (0..scenario.actors())
                .map(|index| scenario.generate_actor(index))
                .collect(),
        );
        let actors = &scenario.initial_actors;
        scenario.config.research = scenario
            .config
            .research
            .take()
            .map(|config| research::ResearchConfig::new(config, actors))
            .transpose()?;
        if let Some(config) = scenario.config.finance.take() {
            let identities = actors.identities();
            let finance = finance::FinanceConfig::new(
                config,
                identities,
                &identities[..scenario.population_rows()],
            )?;
            scenario.totals.cash = add(scenario.totals.cash, finance.initial_cash()?)?;
            scenario.config.finance = Some(finance);
        }
        let railway = std::mem::take(&mut scenario.config.railway);
        scenario.set_railway(railway)?;
        let ring = scenario.ring();
        let side = scenario
            .config
            .map_size
            .div_ceil(scenario.config.search_radius);
        match side <= 256 && ring <= 65 {
            true => Ok(scenario),
            false => Err(anyhow!("scenario exceeds transit or spatial capacity")),
        }
    }

    pub(super) fn set_railway(&mut self, config: railway::RailwayConfig) -> Result<()> {
        self.config.railway =
            railway::RailwayConfig::new(config, &self.initial_actors, self.config.map_size)?;
        self.rail_geography =
            railway::RailGeography::new(&self.config.railway, self.config.property.as_ref())?;
        Ok(())
    }
}

fn generated_totals(
    config: &ScenarioConfig,
    production: &supply::StartingProduction,
    initial_employment: &employment::InitialEmployment,
    retailers: &BTreeMap<u64, enterprise::Distributor>,
) -> Result<InitialTotals> {
    let cash_sources = if config.people.is_some() {
        0
    } else {
        config.population_size()
    };
    let household_cash =
        (1..=u64::from(cash_sources)).try_fold(0u128, |total, household| -> Result<u128> {
            Ok(total + u128::from(initial_employment.cash(config, household as usize - 1)?))
        })?;
    let mut cash = household_cash;
    let food_reserve = supply::StartingReserves::new(config);
    let mut goods = [0; Good::COUNT];
    for stock in food_reserve.household(u64::from(config.population_size())) {
        stock.add_to(&mut goods)?;
    }
    let farmers = employment::InitialSettlement::new(config).farmers(config.population_size());
    if let Some(farm) = supply::StartingFarm::new(config, farmers) {
        for stock in farm.inventory {
            stock.add_to(&mut goods)?;
        }
    }
    for firm in 0..config.firms as usize {
        let good = generated_good(config, firm);
        let retailer = retailers.get(&generated_firm_identity(config, firm));
        cash += u128::from(production.working_cash(config, retailer)?);
        for stock in production.inventory(config, &food_reserve, good, retailer) {
            stock.add_to(&mut goods)?;
        }
    }
    Ok(InitialTotals {
        cash: u64::try_from(cash).context("initial cash exceeds tick capacity")?,
        goods,
    })
}

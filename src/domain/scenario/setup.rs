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
        let initial_employment = employment::InitialEmployment::new(
            &config,
            &production.capacities,
            &production.daily,
            &retailers,
        );
        let totals = match &config.population {
            Some(population) => population.totals()?,
            None => generated_totals(&config, &production, &initial_employment, &retailers)?,
        };
        let resources = resources::ResourceGeography::new(&config)?;
        let mut scenario = Self {
            starting_prices: prices::StartingPrices::configured(&config),
            config,
            totals,
            resources,
            rail_geography: railway::RailGeography::default(),
            production,
            retailers,
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
    let cash = household_cash + u128::from(config.firm_cash) * u128::from(config.firms);
    let food_reserve = supply::StartingReserves::new(config);
    let mut goods = [0; Good::COUNT];
    for stock in food_reserve.household(u64::from(config.population_size())) {
        stock.add_to(&mut goods)?;
    }
    if let enterprise::HouseholdProduction::Available {
        annual_yield_per_plot,
        ..
    } = config.household_production
    {
        let farms =
            u64::from(employment::InitialSettlement::new(config).farmers(config.population_size()));
        goods[Good::Grain as usize - 1] = farms
            * u64::from(annual_yield_per_plot)
            * u64::from(config.demography.initial_household_members.div_ceil(2));
        goods[Good::Tools as usize - 1] = farms;
    }
    for firm in 0..config.firms as usize {
        let good = generated_good(config, firm);
        let retailer = retailers.get(&generated_firm_identity(config, firm));
        for stock in production.inventory(config, &food_reserve, good, retailer) {
            stock.add_to(&mut goods)?;
        }
    }
    Ok(InitialTotals {
        cash: u64::try_from(cash).context("initial cash exceeds tick capacity")?,
        goods,
    })
}

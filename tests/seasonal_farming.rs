use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        enterprise::{HouseholdProduction, LandAccess},
        food::FoodPolicy,
        research::{AdoptionPolicy, ResearchConfig, ResearchConsent, ResearchLab},
        *,
    },
    simulation::Simulation,
};

const CROPS: [Good; 4] = [Good::Grain, Good::Flax, Good::Cotton, Good::Dye];

#[test]
fn generated_industrial_wheat_budget_matches_its_land_and_opening_prices_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(100, 42, Experiment::Baseline)?;
    let state = Simulation::new(scenario.clone(), support::device(64)?)?.download()?;
    let farm = 100;
    assert_eq!(state.actors.at(farm).kind, Good::Grain as u32);
    assert_eq!(state.get(Field::LaborMinutes, farm), 240);
    assert_eq!(state.get(Field::OutputTicks, farm), 20);
    let annual = u64::from(state.actors.at(farm).capacity) * 20 * 365;
    assert_eq!(
        state.get(Field::FarmHectares, farm),
        (annual * 32).div_ceil(1601 * 7)
    );
    assert_eq!(scenario.config().prices, [118, 200, 103]);
    assert!(
        scenario
            .config()
            .distributors
            .iter()
            .any(|store| store.good == Good::Grain)
    );
    Ok(())
}

#[test]
fn industrial_workshop_batches_preserve_unit_inputs_and_saved_scenarios_on_gpu() -> Result<()> {
    let scenario = Scenario::industrial(100, 42, Experiment::Baseline)?;
    let mut legacy = serde_json::to_value(scenario.config())?;
    legacy
        .as_object_mut()
        .unwrap()
        .remove("durable_packet_output");
    let legacy = Scenario::new(serde_json::from_value(legacy)?)?;
    let device = support::device(64)?;
    for (scenario, units) in [(scenario, 1), (legacy, 20)] {
        let state = Simulation::new(scenario, device.clone())?.download()?;
        let workshop = state
            .actors
            .firm_indices()
            .find(|&firm| state.actors.at(firm).kind == Good::Furniture as u32)
            .unwrap();
        assert_eq!(state.get(Field::BasicOutputTicks, workshop), units);
        assert_eq!(state.get(Field::BasicInputTicks, workshop), units);
        assert_eq!(state.get(Field::BasicLaborMinutes, workshop), 20);
    }
    Ok(())
}

fn farming_scenario(home: HouseholdProduction) -> Result<Scenario> {
    let mut config = Scenario::baseline(12, 42, Experiment::Concentrated)?
        .config()
        .clone();
    config.firms = 5;
    config.household_production = home;
    config.entrepreneurship.interested_per_mille = 0;
    config.distribution_rate = 0;
    config.forecast_weight = 0;
    config.behavior.price_response = 0;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    config.demography.mortality.occupational_annual_ppm = [0; Good::COUNT];
    config.resources = ResourceMap::Explicit {
        regions: CROPS
            .into_iter()
            .chain([Good::Wood])
            .map(|good| {
                ResourceRegion::new(
                    good,
                    Position { x: 0, y: 0 },
                    Position { x: 1000, y: 1000 },
                    1000,
                )
            })
            .collect::<Result<Vec<_>>>()?,
    };
    config.population = Some(Population {
        households: (1..=12)
            .map(|identity| {
                let mut inventories = Good::food([0, 0, 100_000]);
                inventories[Good::Wood as usize - 1] = 100_000;
                inventories[Good::Tools as usize - 1] = 1;
                HouseholdSeed {
                    identity,
                    position: Position { x: 500, y: 500 },
                    cash: 10_000,
                    inventories,
                    fatigue: 0,
                }
            })
            .collect(),
        firms: CROPS
            .into_iter()
            .chain([Good::Wood])
            .enumerate()
            .map(|(index, good)| FirmSeed {
                identity: 13 + index as u64,
                position: Position { x: 500, y: 500 },
                owner: 1,
                good,
                cash: 1_000_000,
                inventories: [0; Good::COUNT],
                capacity: 1,
                technique: Technique {
                    labor_minutes: 60,
                    input_ticks: 0,
                    output_ticks: 10,
                },
                wage: 1,
                price: 10,
            })
            .collect(),
    });
    Scenario::new(config)
}

#[test]
fn a_farm_servant_can_work_elsewhere_until_planting_resumes_on_gpu() -> Result<()> {
    let mut config = farming_scenario(HouseholdProduction::Unavailable)?
        .config()
        .clone();
    config.search_radius = 50;
    config.behavior.wage_increment = 0;
    let population = config.population.as_mut().unwrap();
    for household in &mut population.households[1..] {
        household.position.x = 100;
    }
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 58;
    let farm = scenario.index_of(13).unwrap();
    let timber = scenario.index_of(17).unwrap();
    initial.set(Field::Job, 0, farm as u64 + 1);
    initial.set(Field::JobDaysRemaining, 0, 180);
    initial.set(Field::Demand, farm, 100);
    initial.set(Field::Demand, timber, 100);
    let device = support::device(64)?;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, device.clone())?;
    simulation.advance()?;
    let idle = simulation.download()?;
    assert_eq!(idle.get(Field::Plan, farm), 0);
    assert_eq!(idle.get(Field::Employer, 0), timber as u64 + 1);
    assert!(idle.get(Field::GrossIncome, 0) > 0);
    assert_eq!(idle.get(Field::Job, 0), farm as u64 + 1);
    assert_eq!(idle.get(Field::JobDaysRemaining, 0), 179);
    let request = InspectionRequest::new(1, &scenario)?;
    let worker = simulation.inspect(request.clone())?;
    assert_eq!(worker, Inspection::from_state(&scenario, &idle, &request)?);
    assert_eq!(worker.employer, 17);
    assert_eq!(worker.contract_employer, Some(13));
    let mut restored = Simulation::from_checkpoint(simulation.checkpoint()?, device)?;
    restored.advance()?;
    let planting = restored.download()?;
    assert_eq!(planting.get(Field::Employer, 0), farm as u64 + 1);
    assert_eq!(planting.get(Field::JobDaysRemaining, 0), 178);
    Ok(())
}

#[test]
fn crop_firms_accumulate_growing_work_and_release_only_at_harvest_on_gpu() -> Result<()> {
    let scenario = farming_scenario(HouseholdProduction::Unavailable)?;
    let mut initial = scenario.initial()?;
    initial.day = 269;
    for farm in 12..16 {
        initial.set(Field::CropWork, farm, 2110);
        initial.set(Field::Demand, farm, 100);
    }
    let checkpoint = Checkpoint::new(&scenario, &initial)?;
    let mut simulation = Simulation::from_checkpoint(checkpoint, support::device(64)?)?;
    simulation.advance()?;
    let growing = simulation.download()?;
    for farm in 12..16 {
        assert_eq!(growing.get(Field::Produced, farm), 0);
        assert_eq!(growing.get(Field::CropWork, farm), 2120);
    }
    assert_eq!(growing.get(Field::Produced, 16), 10);
    simulation.advance()?;
    let harvested = simulation.download()?;
    for (index, good) in CROPS.into_iter().enumerate() {
        let farm = 12 + index;
        let expected = growing.get(Field::CropWork, farm) * 365 / 211;
        assert_eq!(harvested.get(Field::Produced, farm), expected);
        assert_eq!(harvested.get(Field::inventory(good), farm), expected);
        assert_eq!(harvested.get(Field::CropWork, farm), 0);
    }
    simulation.advance()?;
    let following = simulation.download()?;
    for farm in 12..16 {
        assert_eq!(
            following.get(Field::Produced, farm),
            harvested.get(Field::Produced, farm)
        );
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn wrapped_household_harvest_runs_once_across_new_year_on_gpu() -> Result<()> {
    let scenario = farming_scenario(HouseholdProduction::Available {
        milling_minutes: 30,
        baking_minutes: 30,
        cultivation_minutes: 10,
        harvest_start: 350,
        harvest_days: 30,
        annual_yield_per_plot: 730,
    })?;
    let mut initial = scenario.initial()?;
    initial.day = 364;
    initial.set(Field::LandPlots, 0, 1);
    initial.set(Field::CropWork, 0, 100);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let mut harvested = simulation.download()?;
    assert_eq!(harvested.get(Field::Grain, 0), 10);
    harvested.set(Field::CropWork, 0, 100);
    let mut simulation = Simulation::from_checkpoint(
        Checkpoint::new(&scenario, &harvested)?,
        support::device(64)?,
    )?;
    simulation.advance()?;
    let mut january = simulation.download()?;
    assert_eq!(january.get(Field::Grain, 0), 10);
    assert_eq!(january.get(Field::CropWork, 0), 100);
    january.day = 715;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &january)?, support::device(64)?)?;
    simulation.advance()?;
    let following = simulation.download()?;
    assert_eq!(following.get(Field::Grain, 0), 20);
    assert_eq!(following.get(Field::CropWork, 0), 0);
    Ok(())
}

#[test]
fn wheat_farms_keep_enough_payroll_to_reap_what_they_plant_on_gpu() -> Result<()> {
    let mut config = farming_scenario(HouseholdProduction::Unavailable)?
        .config()
        .clone();
    config.food = Some(FoodPolicy::industrial());
    config.behavior.wage_increment = 0;
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[0].cash = 60;
    firms[1].good = Good::Grain;
    firms[1].cash = 120;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 269;
    for farm in [12, 13] {
        initial.set(Field::CropWork, farm, 20);
        initial.set(Field::Demand, farm, 100);
    }
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let growing = simulation.download()?;
    assert_eq!(growing.get(Field::Plan, 12), 0);
    assert_eq!(growing.get(Field::CropWork, 12), 20);
    assert_eq!(growing.get(Field::CropWork, 13), 30);
    simulation.advance()?;
    let harvest = simulation.download()?;
    for (farm, planted) in [(12, 20), (13, 30)] {
        assert_eq!(harvest.get(Field::Produced, farm), planted * 365 / 211);
        assert_eq!(harvest.get(Field::CropWork, farm), 0);
    }
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn commercial_wheat_needs_harvest_workers_before_the_crop_spoils_on_gpu() -> Result<()> {
    let mut config = farming_scenario(HouseholdProduction::Unavailable)?
        .config()
        .clone();
    config.food = Some(FoodPolicy::industrial());
    let firms = &mut config.population.as_mut().unwrap().firms;
    firms[1].good = Good::Grain;
    firms[1].cash = 0;
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 298;
    for farm in [12, 13] {
        initial.set(Field::CropWork, farm, 200);
    }
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    simulation.advance()?;
    let first = simulation.download()?;
    assert_eq!(first.get(Field::CropWork, 12), 120);
    assert_eq!(
        first.get(Field::Produced, 12),
        (200 * 365 / 211) - (120 * 365 / 211)
    );
    assert_eq!(first.get(Field::CropWork, 13), 200);
    assert_eq!(first.get(Field::Produced, 13), 0);
    let mut resumed = Simulation::from_checkpoint(simulation.checkpoint()?, support::device(64)?)?;
    resumed.advance()?;
    simulation.advance()?;
    let closed = resumed.download()?;
    assert_eq!(closed, simulation.download()?);
    assert_eq!(
        closed.get(Field::Produced, 12),
        (200 * 365 / 211) - (40 * 365 / 211)
    );
    assert_eq!(closed.get(Field::Produced, 13), 0);
    assert_eq!(closed.get(Field::CropWork, 12), 0);
    assert_eq!(closed.get(Field::CropWork, 13), 0);
    Ok(())
}

#[test]
fn threshing_and_consolidation_respect_rotation_and_seed_corn_on_gpu() -> Result<()> {
    let mut config = farming_scenario(HouseholdProduction::Unavailable)?
        .config()
        .clone();
    config.food = Some(FoodPolicy {
        wheat_kg_per_hectare: 320,
        ..FoodPolicy::industrial()
    });
    config.land_access = Some(LandAccess {
        total_plots: 4,
        allocations: Vec::new(),
    });
    config.population.as_mut().unwrap().firms[0].inventories[Good::Tools as usize - 1] = 10;
    config.research = Some(ResearchConfig {
        labs: vec![ResearchLab {
            owner: 13,
            researcher: 1,
            discovered: 1,
            mastered: 1,
            permitted: 1,
            adopted: 1,
            consent: ResearchConsent::Granted,
            production_requirements: 0,
            adoption: AdoptionPolicy::Manual,
            methods: vec![ProductionMethod::MechanicalHarvesting],
            projects: Vec::new(),
        }],
        ..Default::default()
    });
    let scenario = Scenario::new(config)?;
    let device = support::device(64)?;
    let mut initial = Simulation::new(scenario.clone(), device.clone())?.download()?;
    assert_eq!(initial.get(Field::FarmHectares, 12), 4);
    initial.set(Field::FarmHectares, 12, 2);
    initial.set(Field::OccupiedPlots, 12, 2);
    initial.set(Field::CropWork, 12, 400);
    initial.day = 270;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, device)?;
    simulation.advance()?;
    let harvest = simulation.download()?;
    assert_eq!(
        harvest.get(Field::ActiveMethod, 12),
        ProductionMethod::MechanicalHarvesting as u64
    );
    assert_eq!(harvest.get(Field::UnscaledProduced, 12), 276);
    assert_eq!(harvest.get(Field::Produced, 12), 280);
    assert_eq!(harvest.get(Field::HarvestedThisYear, 12), 280);
    assert_eq!(harvest.get(Field::CropWork, 12), 0);
    simulation.advance()?;
    assert_eq!(simulation.download()?.get(Field::Produced, 12), 280);
    simulation.checkpoint()?.restore()?;
    Ok(())
}

#[test]
fn continuous_commercial_wheat_reopens_its_land_budget_once_per_year_on_gpu() -> Result<()> {
    let mut config = farming_scenario(HouseholdProduction::Available {
        milling_minutes: 30,
        baking_minutes: 30,
        cultivation_minutes: 10,
        harvest_start: 0,
        harvest_days: 365,
        annual_yield_per_plot: 730,
    })?
    .config()
    .clone();
    config.food = Some(FoodPolicy {
        wheat_kg_per_hectare: 32,
        ..FoodPolicy::industrial()
    });
    config.land_access = Some(LandAccess {
        total_plots: 1,
        allocations: Vec::new(),
    });
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    initial.day = 364;
    initial.set(Field::Demand, 12, 100);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    for total in [7, 14, 14] {
        simulation.advance()?;
        let state = simulation.download()?;
        assert_eq!(state.get(Field::Produced, 12), total);
        assert_eq!(state.get(Field::HarvestedThisYear, 12), 7);
    }
    assert_eq!(simulation.download()?.get(Field::Plan, 12), 0);
    Ok(())
}

fn household_wheat_market(grain: u64, bread: u64, timber_wage: u32) -> Result<Scenario> {
    let mut config = farming_scenario(HouseholdProduction::Available {
        milling_minutes: 30,
        baking_minutes: 30,
        cultivation_minutes: 10,
        harvest_start: 270,
        harvest_days: 30,
        annual_yield_per_plot: 80,
    })?
    .config()
    .clone();
    config.search_radius = 100;
    let population = config.population.as_mut().unwrap();
    for household in &mut population.households[1..] {
        household.position.x = 0;
    }
    population.households[0].inventories[Good::Grain as usize - 1] = grain;
    population.households[0].inventories[Good::Bread as usize - 1] = bread;
    let mill = &mut population.firms[0];
    mill.good = Good::Flour;
    mill.technique = Technique::new(60, 10, 10)?;
    for firm in &mut population.firms[1..4] {
        firm.position.x = 0;
    }
    population.firms[4].wage = timber_wage;
    Scenario::new(config)
}

fn grain_dealer_scenario(grain: u64, bread: u64) -> Result<Scenario> {
    let mut config = household_wheat_market(grain, bread, 1)?.config().clone();
    let population = config.population.as_mut().unwrap();
    let dealer = &mut population.firms[1];
    dealer.good = Good::Grain;
    dealer.position.x = 500;
    dealer.wage = 2;
    dealer.price = 200;
    population.households[1].position.x = 500;
    config.distributors.push(enterprise::Distributor {
        firm: dealer.identity,
        good: Good::Grain,
        handling_minutes: 5,
        units_per_shift: 24,
        reserve_days: 3,
        margin_per_mille: 250,
    });
    Scenario::new(config)
}

#[test]
fn grain_distributor_buys_household_surplus_and_resells_it_to_a_mill_on_gpu() -> Result<()> {
    let device = support::device(64)?;
    for bread in [100_000, 0] {
        let scenario = grain_dealer_scenario(11, bread)?;
        let mut state = scenario.initial()?;
        let mill = scenario.index_of(13).unwrap();
        let dealer = scenario.index_of(14).unwrap();
        state.set(Field::IdleDays, dealer, 29);
        state.set(Field::Job, 1, dealer as u64 + 1);
        state.set(Field::JobDaysRemaining, 1, 5);
        state.set(Field::LandPlots, 0, 1);
        state.set(Field::Demand, mill, 100);
        let mut simulation =
            Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, device.clone())?;
        for phase in &Phase::ALL[..5] {
            simulation.phase(*phase, bad_econ::simulation::Trace::Discard)?;
        }
        let opening = simulation.download()?;
        simulation.phase(
            Phase::ProvisionPurchase,
            bad_econ::simulation::Trace::Record,
        )?;
        let cleared = simulation.download()?;
        let sold = cleared.get(Field::GrainSales, 0);
        assert_eq!(sold, u64::from(bread > 0));
        assert_eq!(
            cleared.get(Field::Grain, 0) + sold,
            opening.get(Field::Grain, 0)
        );
        assert_eq!(cleared.get(Field::Transit, dealer), sold);
        assert_eq!(cleared.get(Field::Transit, mill), 0);
        if bread > 0 {
            assert_eq!(cleared.get(Field::GrainReserve, 0), 10);
            assert_eq!(cleared.get(Field::Grain, 0), 10);
            assert_eq!(
                cleared.get(Field::Cash, 0) - opening.get(Field::Cash, 0),
                sold * u64::from(scenario.config().prices[0])
            );
            assert_eq!(
                cleared.get(Field::ReplacementCost, dealer),
                u64::from(scenario.config().prices[0])
            );
        } else {
            assert!(cleared.get(Field::GrainReserve, 0) > 10);
        }
        for phase in &Phase::ALL[6..] {
            simulation.phase(*phase, bad_econ::simulation::Trace::Discard)?;
        }
        let forecast = simulation.download()?.get(Field::ExpectedGrainSales, 0);
        assert_eq!(forecast, sold * 365 / 30);
        let mut restored = Simulation::from_checkpoint(simulation.checkpoint()?, device.clone())?;
        simulation.advance()?;
        restored.advance()?;
        let resale = simulation.download()?;
        assert_eq!(restored.download()?, resale);
        assert_eq!(resale.get(Field::Sales, dealer), sold);
        assert_eq!(resale.get(Field::Transit, mill), sold);
        if sold > 0 {
            assert_eq!(resale.get(Field::FirstInputCost, mill), 200);
        }
    }
    Ok(())
}

#[test]
fn grain_distributor_orders_only_what_its_staff_can_handle_on_gpu() -> Result<()> {
    let scenario = grain_dealer_scenario(100_000, 100_000)?;
    let mut state = scenario.initial()?;
    let dealer = scenario.index_of(14).unwrap();
    state.set(Field::Demand, dealer, 100_000);
    let daily = u64::from(state.actors.at(dealer).capacity) * state.get(Field::OutputTicks, dealer);
    let coverage = state.get(Field::StockCoverage, dealer) + scenario.ring() as u64;
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    let cleared = simulation.download()?;
    assert_eq!(cleared.get(Field::GrainSales, 0), daily * coverage);
    assert_eq!(cleared.get(Field::Transit, dealer), daily * coverage);
    Ok(())
}

#[test]
fn a_mill_cannot_buy_household_grain_without_a_distributor_on_gpu() -> Result<()> {
    let scenario = household_wheat_market(1000, 100_000, 1)?;
    let mut state = scenario.initial()?;
    let mill = scenario.index_of(13).unwrap();
    state.set(Field::LandPlots, 0, 1);
    state.set(Field::Demand, mill, 100);
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    simulation.advance()?;
    let after = simulation.download()?;
    assert!(after.get(Field::Grain, 0) > after.get(Field::GrainReserve, 0));
    assert_eq!(after.get(Field::GrainSales, 0), 0);
    assert_eq!(after.get(Field::Transit, mill), 0);
    Ok(())
}

#[test]
fn planted_wheat_responds_to_carryover_sales_and_paid_work_on_gpu() -> Result<()> {
    let device = support::device(64)?;
    for (stock, demand, wage, planting) in [
        (40, 0, 1, 0),
        (40, 365, 1, 4),
        (900, 365, 1, 0),
        (40, 365, 100, 0),
    ] {
        let scenario = household_wheat_market(stock, 100_000, wage)?;
        let mut state = scenario.initial()?;
        state.day = 100;
        let timber = scenario.index_of(17).unwrap();
        state.set(Field::Job, 0, timber as u64 + 1);
        state.set(Field::JobDaysRemaining, 0, 10);
        state.set(Field::LandPlots, 0, 1);
        state.set(Field::ExpectedGrainSales, 0, demand);
        state.set(Field::Demand, timber, 100);
        let mut simulation =
            Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, device.clone())?;
        for phase in &Phase::ALL[..5] {
            simulation.phase(*phase, bad_econ::simulation::Trace::Discard)?;
        }
        let planted = simulation.download()?;
        assert_eq!(planted.get(Field::Employer, 0), timber as u64 + 1);
        assert_eq!(planted.get(Field::CropWork, 0), planting);
        assert_eq!(planted.get(Field::Grain, 0), stock - planting.div_ceil(80));
    }
    Ok(())
}

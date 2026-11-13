#![cfg(feature = "compute")]
use super::support;
use anyhow::Result;
use bad_econ::{
    domain::{
        Checkpoint, Experiment, Inspection, InspectionRequest, ObservationSnapshot, Scenario,
        calendar::{CalendarDate, HarvestSeason, Month, Season, Weather},
        enterprise::HouseholdProduction,
    },
    runtime::SimulationCommand,
    simulation::Simulation,
};

#[test]
fn dates_weather_and_year_rollover_survive_gpu_checkpoints() -> Result<()> {
    assert_eq!(CalendarDate::on_day(0).to_string(), "01 Jan 0001");
    assert_eq!(CalendarDate::on_day(58).to_string(), "28 Feb 0001");
    assert_eq!(CalendarDate::on_day(59).to_string(), "01 Mar 0001");
    assert_eq!(CalendarDate::on_day(364).to_string(), "31 Dec 0001");
    assert_eq!(CalendarDate::on_day(365).to_string(), "01 Jan 0002");
    assert_eq!(CalendarDate::on_day(730).to_string(), "01 Jan 0003");
    assert_eq!(CalendarDate::on_day(365).year(), 2);
    assert_eq!(CalendarDate::on_day(365).month(), Month::January);
    assert_eq!(CalendarDate::on_day(365).day(), 1);
    assert_eq!(CalendarDate::on_day(365).day_of_year(), 0);
    assert!(CalendarDate::new(0, Month::January, 1).is_err());
    assert!(CalendarDate::new(1, Month::January, 0).is_err());
    assert!(CalendarDate::new(1, Month::February, 29).is_err());
    assert!(Month::from_number(13).is_err());
    assert!(
        serde_json::from_str::<CalendarDate>(r#"{"year":1,"month":"February","day":29}"#).is_err()
    );
    assert_eq!(Weather::on_day(58).season, Season::Winter);
    assert_eq!(Weather::on_day(59).season, Season::Spring);
    assert_eq!(Weather::on_day(151).season, Season::Summer);
    assert_eq!(Weather::on_day(243).season, Season::Autumn);
    assert_eq!(Weather::on_day(334).season, Season::Winter);
    assert_eq!(Weather::on_day(269).harvest, HarvestSeason::Growing);
    assert_eq!(Weather::on_day(270).harvest, HarvestSeason::Harvesting);
    assert_eq!(Weather::on_day(299).harvest, HarvestSeason::Harvesting);
    assert_eq!(Weather::on_day(300).harvest, HarvestSeason::Dormant);
    assert_eq!(Weather::on_day(0).temperature_celsius, -5);
    assert_eq!(Weather::on_day(0).heating_units_per_person, 3);
    assert_eq!(Weather::on_day(182).temperature_celsius, 23);
    assert_eq!(Weather::on_day(182).heating_units_per_person, 0);
    assert_eq!(Weather::on_day(365).temperature_celsius, -5);

    let mut config = Scenario::baseline(30, 42, Experiment::Baseline)?
        .config()
        .clone();
    config.household_production = HouseholdProduction::Available {
        milling_minutes: 30,
        baking_minutes: 30,
        cultivation_minutes: 10,
        harvest_start: 360,
        harvest_days: 10,
        annual_yield_per_plot: 730,
    };
    let scenario = Scenario::new(config)?;
    assert_eq!(scenario.weather(148).harvest, HarvestSeason::Dormant);
    assert_eq!(scenario.weather(149).harvest, HarvestSeason::Growing);
    assert_eq!(scenario.weather(359).harvest, HarvestSeason::Growing);
    assert_eq!(scenario.weather(360).harvest, HarvestSeason::Harvesting);
    assert_eq!(scenario.weather(369).harvest, HarvestSeason::Harvesting);
    assert_eq!(scenario.weather(370).harvest, HarvestSeason::Dormant);
    let mut state = scenario.initial()?;
    state.day = 364;
    assert_eq!(
        ObservationSnapshot::from_state(&scenario, &state)?.weather,
        scenario.weather(364)
    );
    let mut simulation =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &state)?, support::device(64)?)?;
    assert_eq!(simulation.snapshot.weather, scenario.weather(364));
    assert_eq!(
        simulation.snapshot.weather.harvest,
        HarvestSeason::Harvesting
    );
    simulation.advance()?;
    assert_eq!(simulation.snapshot.day, 365);
    assert_eq!(simulation.snapshot.weather, scenario.weather(365));
    let request = InspectionRequest::new(1, &scenario)?;
    let inspection = simulation.inspect(request.clone())?;
    assert_eq!(inspection.weather, simulation.snapshot.weather);
    assert_eq!(
        inspection.weather,
        Inspection::from_state(&scenario, &simulation.download()?, &request)?.weather
    );
    let checkpoint = simulation.checkpoint()?;
    let checkpoint: Checkpoint = serde_json::from_str(&serde_json::to_string(&checkpoint)?)?;
    simulation.advance()?;
    let expected = simulation.download()?;
    let expected_weather = simulation.snapshot.weather;
    simulation.command(SimulationCommand::Restore(Box::new(checkpoint)))?;
    assert_eq!(simulation.snapshot.weather, scenario.weather(365));
    simulation.advance()?;
    assert_eq!(simulation.download()?, expected);
    assert_eq!(simulation.snapshot.weather, expected_weather);
    Ok(())
}

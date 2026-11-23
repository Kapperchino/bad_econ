use super::support;
use anyhow::Result;
use bad_econ::domain::*;

fn scenario() -> Result<Scenario> {
    let mut config = Scenario::baseline(3, 19, Experiment::Baseline)?
        .config()
        .clone();
    config.distribution_rate = 0;
    config.behavior = BehaviorPolicy {
        work_fatigue: 13,
        rest_recovery: 17,
        wage_increment: 0,
        price_response: 0,
    };
    config.resources = ResourceMap::Explicit {
        regions: vec![ResourceRegion::new(
            Good::Grain,
            Position { x: 0, y: 0 },
            Position {
                x: config.map_size,
                y: config.map_size,
            },
            config.map_size,
        )?],
    };
    config.population = Some(Population {
        households: [901, 101, 303]
            .into_iter()
            .map(|identity| HouseholdSeed {
                identity,
                position: Position { x: 0, y: 0 },
                cash: 1000,
                inventories: Good::food([0, 0, 10]),
                fatigue: 0,
            })
            .collect(),
        firms: vec![
            FirmSeed {
                identity: 8000,
                owner: 303,
                good: Good::Bread,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: Good::food([0, 3, 0]),
                capacity: 1,
                technique: Technique::new(240, 3, 2)?,
                wage: 3,
                price: 30,
            },
            FirmSeed {
                identity: 2000,
                owner: 901,
                good: Good::Grain,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: Good::food([0, 0, 0]),
                capacity: 1,
                technique: Technique::new(120, 0, 5)?,
                wage: 1,
                price: 10,
            },
            FirmSeed {
                identity: 5000,
                owner: 101,
                good: Good::Flour,
                cash: 10_000,
                position: Position { x: 0, y: 0 },
                inventories: Good::food([2, 0, 0]),
                capacity: 1,
                technique: Technique::new(180, 2, 3)?,
                wage: 2,
                price: 20,
            },
        ],
    });
    Scenario::new(config)
}

#[test]
fn actor_rows_round_trip_on_gpu_and_reject_mismatched_lengths() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let scenario = scenario()?;
    let expected = scenario.initial()?;
    let device = support::device(64)?;
    let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
    assert_eq!(simulation.download()?, expected);
    let encoded = serde_json::to_value(&expected)?;
    assert_eq!(
        encoded["actors"]["identity"],
        serde_json::json!([101, 303, 901, 2000, 5000, 8000])
    );
    for field in ["identity", "x", "y", "kind", "owner", "capacity"] {
        let mut invalid = encoded.clone();
        invalid["actors"][field].as_array_mut().unwrap().pop();
        assert!(serde_json::from_value::<State>(invalid).is_err());
    }
    for actor in expected.actors.iter() {
        let request = InspectionRequest::new(actor.identity, &scenario)?;
        assert_eq!(simulation.inspect(request.clone())?.actor, actor);
        assert_eq!(
            Inspection::from_state(&scenario, &expected, &request)?.actor,
            actor
        );
    }
    let checkpoint = simulation.checkpoint()?;
    let mut old = serde_json::to_value(&checkpoint)?;
    old["version"] = serde_json::json!(2);
    assert!(serde_json::from_value::<Checkpoint>(old).is_err());
    let check_rows = |rows: &ActorRows| -> Result<()> {
        let encoded = serde_json::to_value(rows)?;
        assert_eq!(encoded.as_object().unwrap().len(), 6);
        let decoded: ActorRows = serde_json::from_value(encoded)?;
        assert_eq!(&decoded, rows);
        assert_eq!(rows.iter().collect::<ActorRows>(), *rows);
        let words = |rows: &ActorRows| {
            rows.iter()
                .flat_map(|actor| {
                    [
                        actor.identity as u32,
                        (actor.identity >> 32) as u32,
                        actor.position.x,
                        actor.position.y,
                        actor.kind,
                        actor.owner,
                        actor.capacity,
                    ]
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(device.read(&device.tensor(&words(rows)))?, words(&decoded));
        Ok(())
    };
    let mut rows = expected.actors.clone();
    for (index, kind) in [
        (0, Good::Bread as u32),
        (3, 0),
        (0, 0),
        (4, 0),
        (5, 0),
        (2, Good::Bread as u32),
        (2, 0),
        (5, Good::Bread as u32),
        (4, Good::Bread as u32),
        (4, 0),
        (5, 0),
    ] {
        let mut actor = rows.at(index);
        actor.kind = kind;
        rows.set(index, actor);
        check_rows(&rows)?;
    }
    for kind in [0, Good::Bread as u32, 0, Good::Tools as u32] {
        let actor = Actor {
            identity: 10_000 + rows.len() as u64,
            position: Position { x: 1, y: 2 },
            kind,
            owner: 0,
            capacity: 1,
        };
        rows = rows.iter().chain(std::iter::once(actor)).collect();
        check_rows(&rows)?;
    }
    Ok(())
}

#[test]
fn worker_cohorts_settle_multiple_shifts_and_replay_on_gpu() -> Result<()> {
    worker_shifts(None)
}

#[test]
fn negotiated_cohort_shifts_pay_each_hire_on_gpu() -> Result<()> {
    worker_shifts(Some(bargaining::BargainingPolicy {
        legal_minimum: 1,
        quantum: 1,
        daily_cash_cost: 0,
        funded_other: 0,
        participating_adults: 1,
        normal_day_minutes: 480,
        work_cost_per_minute: 0,
        desired_wage: 1,
        mandate: 0,
        withdrawal: 0,
        alternatives: 0,
        concentration: 0,
        replacements: 0,
    }))
}

fn worker_shifts(bargaining: Option<bargaining::BargainingPolicy>) -> Result<()> {
    use bad_econ::simulation::Simulation;
    let scenario = Scenario::for_people(
        29,
        CohortSizes::default(),
        Economy::Agrarian,
        42,
        Experiment::Concentrated,
    )?;
    let mut config = scenario.config().clone();
    config.buffer_days = 1;
    config.behavior.work_fatigue = 48;
    config.behavior.rest_recovery = 2;
    config.bargaining = bargaining;
    config.payroll = Some(PayrollPolicy {
        base_rate: 0,
        employee_charge: 0,
        brackets: vec![TaxBracket {
            threshold: 100,
            marginal_rate: 500,
        }],
    });
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    assert_eq!(initial.initial_population, 29);
    assert_eq!(scenario.population_rows(), 6);
    assert_eq!(
        initial
            .actors
            .iter()
            .take(6)
            .map(|actor| actor.capacity)
            .collect::<Vec<_>>(),
        [1, 1, 1, 10, 10, 6]
    );
    for firm in initial.actors.iter().filter(|actor| actor.kind > 0) {
        assert_eq!(initial.actors.at(firm.owner as usize).capacity, 1);
        assert_eq!(
            scenario.index_of(firm.identity),
            Some(firm.identity as usize - 30 + 6)
        );
    }
    let encoded = serde_json::to_value(scenario.config())?;
    assert_eq!(encoded["people"], 29);
    assert!(encoded.get("households").is_none());
    assert_eq!(Scenario::new(serde_json::from_value(encoded)?)?, scenario);
    let device = support::device(64)?;
    let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
    simulation.advance()?;
    let state = simulation.download()?;
    assert!(
        state
            .actors
            .iter()
            .enumerate()
            .any(|(row, _)| state.get(Field::Employed, row) > 1)
    );
    assert!(state.people.is_empty());
    for row in 0..scenario.population_rows() {
        assert!(state.get(Field::Employed, row) <= state.get(Field::EligibleWorkers, row));
        let exertion: u64 = [
            Field::Hours,
            Field::HomeMinutes,
            Field::CartageMinutes,
            Field::ResearchMinutes,
        ]
        .into_iter()
        .map(|field| state.get(field, row))
        .sum();
        let minutes = exertion / state.get(Field::EligibleWorkers, row).max(1);
        let fatigue = (initial.get(Field::Fatigue, row) + minutes * 48 / 480)
            .saturating_sub(2)
            .min(1000);
        assert_eq!(state.get(Field::Fatigue, row), fatigue);
        let gross = state.get(Field::GrossIncome, row);
        let allowance = 100 * u64::from(state.actors.at(row).capacity);
        assert_eq!(
            state.get(Field::TaxTreasury, row),
            gross.saturating_sub(allowance) / 2
        );
        assert_eq!(
            state.get(Field::GrossIncome, row),
            state.get(Field::Hours, row) * u64::from(scenario.config().wage)
        );
    }
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    let mut restored = Simulation::from_checkpoint(simulation.checkpoint()?, device)?;
    restored.advance()?;
    simulation.advance()?;
    assert_eq!(
        restored.download()?.digest(),
        simulation.download()?.digest()
    );
    Ok(())
}

#[test]
fn farmer_and_worker_sizes_preserve_people_and_opening_resources_on_gpu() -> Result<()> {
    use bad_econ::simulation::Simulation;
    use std::num::NonZeroU32;
    let make = |workers, farmers| {
        Scenario::for_people(
            1001,
            CohortSizes {
                workers: NonZeroU32::new(workers).unwrap(),
                farmers: NonZeroU32::new(farmers).unwrap(),
            },
            Economy::Industrial,
            42,
            Experiment::Concentrated,
        )
    };
    let individual = make(1, 1)?;
    let grouped = make(4, 7)?;
    let original = individual.initial()?;
    let initial = grouped.initial()?;
    assert_eq!(original.initial_cash, initial.initial_cash);
    assert_eq!(original.initial_goods, initial.initial_goods);
    assert_eq!(initial.initial_population, 1001);
    assert_eq!(original.ownership, initial.ownership);
    let jobs = |state: &State| {
        let mut counts = std::collections::BTreeMap::new();
        for (row, actor) in state
            .actors
            .iter()
            .enumerate()
            .filter(|(_, actor)| actor.kind == 0)
        {
            if let Some(firm) = state.get(Field::Job, row).checked_sub(1) {
                let remaining = state.get(Field::JobDaysRemaining, row);
                *counts
                    .entry((state.actors.at(firm as usize).identity, remaining))
                    .or_insert(0u32) += actor.capacity;
            }
        }
        counts
    };
    let opening_jobs = jobs(&initial);
    assert_eq!(jobs(&original), opening_jobs);
    assert!(
        opening_jobs
            .keys()
            .any(|&(_, days)| (1..14).contains(&days))
    );
    assert!(opening_jobs.keys().any(|&(_, days)| days > 30));
    for row in 0..grouped.population_rows() {
        let actor = initial.actors.at(row);
        let bound = if initial.get(Field::LandPlots, row) > 0 {
            7
        } else {
            4
        };
        assert!((1..=bound).contains(&actor.capacity));
    }
    assert!(
        initial
            .actors
            .iter()
            .any(|actor| actor.kind == 0 && actor.capacity == 7)
    );
    assert!(
        initial
            .actors
            .iter()
            .any(|actor| actor.kind == 0 && actor.capacity == 4)
    );
    let mut sim = Simulation::new(grouped.clone(), support::device(64)?)?;
    sim.advance()?;
    let audited = sim.download()?;
    assert_eq!(
        sim.snapshot,
        ObservationSnapshot::from_state(&grouped, &audited)?
    );
    sim.checkpoint()?;
    Ok(())
}

#[test]
fn a_founder_buys_materials_and_leaves_an_explicit_cohort_on_gpu() -> Result<()> {
    use bad_econ::simulation::Simulation;
    let mut config = Scenario::for_people(
        11,
        CohortSizes::default(),
        Economy::Agrarian,
        42,
        Experiment::Concentrated,
    )?
    .config()
    .clone();
    config.household_production = enterprise::HouseholdProduction::industrial();
    config.entrepreneurship.interested_per_mille = 1000;
    config.demography.annual_births_per_million_fertile_women = 0;
    config.demography.mortality.background_annual_ppm = [0; 5];
    let position = Position { x: 500, y: 500 };
    let seeds = (1..=11)
        .map(|identity| {
            let mut inventories = Good::food([0, 0, 1000]);
            inventories[Good::Wood as usize - 1] = 100;
            HouseholdSeed {
                identity,
                position,
                cash: 1_000_000,
                inventories,
                fatigue: identity as u32,
            }
        })
        .collect();
    let firms = [Good::Grain, Good::Flour, Good::Bread, Good::Tools]
        .into_iter()
        .enumerate()
        .map(|(index, good)| {
            let mut stock = Good::food([1000, 1000, 1000]);
            stock[good as usize - 1] = 1000;
            Ok(FirmSeed {
                identity: 100 + index as u64,
                owner: 11,
                position,
                good,
                cash: 100_000,
                capacity: 4,
                inventories: stock,
                technique: Technique::new(30, u32::from(good != Good::Grain), 10)?,
                wage: 1,
                price: if good == Good::Bread { 500 } else { 10 },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    config.firms = firms.len() as u32;
    config.population = Some(Population {
        households: seeds,
        firms,
    });
    let scenario = Scenario::new(config)?;
    let mut initial = scenario.initial()?;
    assert_eq!(scenario.population_rows(), 2);
    assert_eq!(initial.actors.at(0).capacity, 10);
    assert_eq!(initial.get(Field::Cash, 0), 10_000_000);
    assert!(initial.people.is_empty());
    assert_eq!(initial.get(Field::YoungAdults, 0), 10);
    assert_eq!(initial.get(Field::Fatigue, 0), 5);
    initial.set(Field::InvestmentGood, 0, 1);
    let mut sim =
        Simulation::from_checkpoint(Checkpoint::new(&scenario, &initial)?, support::device(64)?)?;
    sim.advance()?;
    let ordered = sim.download()?;
    assert_eq!(ordered.get(Field::Tools, 0), 0);
    assert_eq!(
        ordered.values[(TRANSIT_FIELDS + Good::Tools as usize - 1) * ordered.actors.len()],
        20
    );
    sim.advance()?;
    let separated = sim.download()?;
    let founder = separated
        .actors
        .iter()
        .enumerate()
        .find(|(row, actor)| actor.kind == 0 && separated.get(Field::ParentHousehold, *row) == 1)
        .expect("founder separated from the cohort")
        .0;
    assert_eq!(separated.get(Field::LivingMembers, founder), 1);
    assert_eq!(separated.get(Field::LivingMembers, 0), 9);
    sim.advance()?;
    let state = sim.download()?;
    assert!(
        state
            .actors
            .iter()
            .any(|firm| firm.kind > 0 && firm.owner as usize == founder)
    );
    assert_eq!(state.get(Field::InvestmentFirm, 0), 0);
    assert_eq!(state.people.len(), initial.people.len());
    sim.checkpoint()?.restore()?;
    Ok(())
}

fn legacy_fields(state: &mut State, scenario: &Scenario, missing: std::ops::Range<usize>) {
    let rows = state.actors.len();
    let end = scenario.fields() * rows;
    state.values.drain(end - RetailAccount::FIELDS * rows..end);
    if missing.end == USED_FIELDS {
        let shipments = Good::COUNT * scenario.ring();
        state
            .values
            .drain((BASE_FIELDS + shipments) * rows..(BASE_FIELDS + 2 * shipments) * rows);
    }
    state.values.drain(missing.start * rows..missing.end * rows);
}

#[test]
fn legacy_checkpoints_migrate_and_resume_on_gpu() -> Result<()> {
    use bad_econ::simulation::Simulation;
    use sha2::{Digest, Sha256};

    let device = support::device(64)?;
    let scenario = scenario()?;
    let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
    simulation.advance()?;
    let state = simulation.download()?;
    let checkpoint = serde_json::to_value(simulation.checkpoint()?)?;
    for (version, missing) in [
        (28, 0..0),
        (23, Field::JobDaysRemaining as usize..USED_FIELDS),
        (24, Field::EligibleWorkers as usize..USED_FIELDS),
        (25, Field::Infants as usize..USED_FIELDS),
        (26, Field::GrainReserve as usize..USED_FIELDS),
        (27, Field::RoadFreight as usize..USED_FIELDS),
        (
            27,
            Field::GrainReserve as usize..Field::RoadFreight as usize,
        ),
    ] {
        let mut old_state = state.clone();
        legacy_fields(&mut old_state, &scenario, missing);
        let checksum = bad_econ::arithmetic::hex(&Sha256::digest(serde_json::to_vec(&(
            scenario.config(),
            &old_state,
        ))?));
        let mut legacy = checkpoint.clone();
        legacy["state"] = serde_json::to_value(old_state)?;
        legacy["version"] = version.into();
        assert!(serde_json::from_value::<Checkpoint>(legacy.clone()).is_err());
        legacy["checksum"] = checksum.into();
        let mut restored =
            Simulation::from_checkpoint(serde_json::from_value(legacy)?, device.clone())?;
        restored.advance()?;
        let migrated = restored.download()?;
        assert_eq!(
            restored.snapshot,
            ObservationSnapshot::from_state(&scenario, &migrated)?
        );
    }
    Ok(())
}

#[test]
fn individual_checkpoint_members_resume_as_cohort_counts_on_gpu() -> Result<()> {
    use bad_econ::{domain::demography::*, simulation::Simulation};
    use sha2::{Digest, Sha256};
    let start = Scenario::for_people(
        29,
        CohortSizes::default(),
        Economy::Agrarian,
        42,
        Experiment::Concentrated,
    )?;
    let mut config = start.config().clone();
    config.demography.people = (1..=29)
        .map(|id| PersonSeed {
            identity: id + 1000,
            household: id,
            age_days: match id {
                1..=5 => 0,
                6..=12 => 70 * DAYS_PER_YEAR,
                _ => 24 * DAYS_PER_YEAR,
            },
            sex: Sex::Male,
        })
        .collect();
    let scenario = Scenario::new(config)?;
    let initial = scenario.initial()?;
    let mut legacy = initial.clone();
    legacy_fields(&mut legacy, &scenario, Field::Infants as usize..USED_FIELDS);
    legacy.next_identity = 1030;
    legacy.people = scenario
        .config()
        .demography
        .people
        .iter()
        .map(|seed| {
            let household = match seed.household {
                1..=3 => seed.household,
                id => 4 + ((id - 4) / 10) * 10,
            };
            Person {
                identity: seed.identity,
                household,
                age_days: seed.age_days,
                sex: seed.sex,
                state: LivingState::Living,
                employer: None,
                work_minutes: 0,
                fatigue: 0,
                famine_exposure: 0,
                cold_exposure: 0,
                birth_cooldown: 0,
            }
        })
        .collect();
    let checksum = bad_econ::arithmetic::hex(&Sha256::digest(serde_json::to_vec(&(
        scenario.config(),
        &legacy,
    ))?));
    let mut document = serde_json::to_value(Checkpoint::new(&scenario, &initial)?)?;
    document["state"] = serde_json::to_value(legacy)?;
    document["version"] = 25.into();
    document["checksum"] = checksum.into();
    let mut sim =
        Simulation::from_checkpoint(serde_json::from_value(document)?, support::device(64)?)?;
    assert_eq!(sim.snapshot.population.living, 29);
    assert_eq!(sim.snapshot.population.dependents, 12);
    assert!(sim.download()?.people.is_empty());
    sim.advance()?;
    assert_eq!(sim.snapshot.population.births, 0);
    let state = sim.download()?;
    assert_eq!(
        sim.snapshot,
        ObservationSnapshot::from_state(&scenario, &state)?
    );
    Ok(())
}

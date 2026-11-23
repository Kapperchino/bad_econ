use super::{fixtures::WoodFirms, support};

use anyhow::{Context, Result};
use bad_econ::{
    domain::{finance::*, research::*, *},
    runtime::SimulationCommand,
    simulation::{Simulation, Trace},
};

fn configuration() -> Result<ScenarioConfig> {
    WoodFirms {
        seed: 42,
        firm_cash: 0,
        labor_minutes: 80,
    }
    .configuration()
}

fn finance() -> FinanceConfig {
    FinanceConfig {
        banks: vec![BankSeed {
            identity: 100,
            reserves: 100,
            minimum_capital: 100,
            risk_weight: 1000,
        }],
        loans: vec![LoanContract {
            identity: 200,
            bank: 100,
            borrower: 4,
            requested: 80,
            target_cash: 80,
            first_day: 0,
            term: 2,
            daily_rate: 0,
            surplus: 100,
            debt_service: 0,
            service_fraction: 1000,
            advance: 1000,
            collateral: 100,
            secured: 0,
            consent: Consent::Granted,
        }],
        support: Some(SupportPolicy {
            treasury: 6,
            recipients: [2, 4, 6]
                .into_iter()
                .enumerate()
                .map(|(index, target_cash)| SupportRecipient {
                    household: index as u64 + 1,
                    target_cash,
                    consent: Consent::Granted,
                })
                .collect(),
        }),
    }
}

#[test]
fn bank_credit_funds_real_production_and_support_uses_real_household_cash() -> Result<()> {
    let mut config = configuration()?;
    config.finance = Some(finance());
    config.payroll = Some(PayrollPolicy {
        base_rate: 100,
        employee_charge: 50,
        brackets: vec![],
    });
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    assert_eq!(simulation.snapshot.cash, 106);
    simulation.advance()?;
    let state = simulation.download()?;
    assert_eq!(state.get(Field::Produced, 3), 1);
    let worker = (0..3)
        .find(|&household| state.get(Field::Employed, household) > 0)
        .context("funded work must pay a household")?;
    assert_eq!(state.get(Field::Cash, worker), 68);
    assert_eq!(state.get(Field::GrossIncome, worker), 80);
    assert_eq!(
        (0..3)
            .filter(|&household| household != worker)
            .map(|household| state.get(Field::Cash, household))
            .sum::<u64>(),
        6
    );
    assert_eq!(simulation.snapshot.financial_cash, 20);
    assert_eq!(simulation.snapshot.loan_principal, 80);
    assert_eq!(simulation.snapshot.active_loans, 1);
    assert_eq!(simulation.snapshot.cash, 106);
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&simulation.scenario, &state)?
    );
    let checkpoint = simulation.checkpoint()?;
    (0..3).try_for_each(|_| simulation.advance())?;
    let expected = simulation.download()?;
    assert_eq!(simulation.snapshot.overdue_loans, 1);
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    (0..3).try_for_each(|_| simulation.advance())?;
    assert_eq!(simulation.download()?, expected);
    Ok(())
}

fn research_configuration() -> Result<Scenario> {
    let mut config = configuration()?;
    let firm = &mut config.population.as_mut().unwrap().firms[0];
    firm.good = Good::Flour;
    firm.inventories[0] = 20;
    firm.cash = 1000;
    firm.technique = Technique {
        labor_minutes: 80,
        input_ticks: 1,
        output_ticks: 1,
    };
    config.research = Some(ResearchConfig {
        labs: vec![ResearchLab {
            methods: Vec::new(),
            owner: 4,
            researcher: 1,
            discovered: 0,
            mastered: 0,
            permitted: 1,
            adopted: 0,
            adoption: AdoptionPolicy::MasterAndAdoptPermitted,
            consent: ResearchConsent::Granted,
            production_requirements: 1,
            projects: vec![ResearchProject {
                identity: 300,
                prerequisites: 0,
                outcomes: 1,
                inputs: Good::food([1, 0, 0]),
                labor_minutes: 16,
                minimum_effort: 1,
                pioneer_effort: 1,
                capacity: 1000,
                relatedness: 1000,
                shape: 1,
                exposure: 0,
                exposure_signal: 1000,
            }],
        }],
        equipment: vec![Equipment {
            firm: 4,
            remaining_service: 2,
            condition: 1000,
            nameplate: 1,
            downtime: 0,
            wear_per_packet: 0,
            maintenance_good: Good::Grain,
            maintenance_ticks: 0,
            restoration_per_tick: 0,
        }],
        natural_stocks: vec![NaturalStock {
            firm: 4,
            stock: 4,
            capacity: 4,
            renewal_rate: 0,
            extraction_per_packet: 1,
        }],
        strikes: vec![StrikePolicy {
            household: 3,
            grievance: 1000,
            solidarity: 1000,
            trust: 1000,
            efficacy: 1000,
            peers: 1000,
            cost: 0,
            risk: 0,
        }],
    });
    Scenario::new(config)
}

#[test]
fn research_adoption_equipment_and_natural_stocks_constrain_real_production() -> Result<()> {
    let mut simulation = Simulation::new(research_configuration()?, support::device(64)?)?;
    simulation.advance()?;
    assert_eq!(simulation.snapshot.produced[1], 0);
    assert_eq!(simulation.snapshot.research.attempts, 1);
    assert_eq!(simulation.snapshot.research.discoveries, 1);
    assert_eq!(simulation.snapshot.research.consumed, Good::food([1, 0, 0]));
    let checkpoint = simulation.checkpoint()?;
    for phase in Phase::ALL
        .into_iter()
        .take_while(|phase| *phase != Phase::AccountsDecisions)
    {
        simulation.phase(phase, Trace::Discard)?;
    }
    assert_eq!(simulation.checkpoint()?, checkpoint);
    simulation.command(SimulationCommand::Restore(checkpoint.clone().into()))?;
    (0..4).try_for_each(|_| simulation.advance())?;
    assert_eq!(simulation.snapshot.produced[1], 2);
    assert!(simulation.snapshot.research.striking <= 1);
    assert_eq!(simulation.snapshot.research.remaining_service, 0);
    assert_eq!(simulation.snapshot.research.natural_stock, 2);
    let expected = simulation.download()?;
    assert_eq!(
        simulation.snapshot,
        ObservationSnapshot::from_state(&simulation.scenario, &expected)?
    );
    simulation.command(SimulationCommand::Restore(checkpoint.into()))?;
    (0..4).try_for_each(|_| simulation.advance())?;
    assert_eq!(simulation.download()?, expected);
    Ok(())
}

#[test]
fn research_snapshot_reads_mixed_width_columns_on_gpu() -> Result<()> {
    let mut config = research_configuration()?.config().clone();
    let research = config.research.as_mut().context("research configuration")?;
    let projects = &mut research.labs[0].projects;
    projects.push(ResearchProject {
        identity: projects[0].identity + 1,
        exposure: 73,
        ..projects[0].clone()
    });
    let service = u64::from(u32::MAX) + 19;
    research.equipment[0].remaining_service = service;
    let mut simulation = Simulation::new(Scenario::new(config)?, support::device(64)?)?;
    let mut snapshot = simulation
        .download()?
        .research
        .context("research snapshot")?;
    let research = simulation
        .scenario
        .config()
        .research
        .as_ref()
        .context("research configuration")?;
    assert_eq!(snapshot.lab(research, 0)?.permitted, 1);
    assert_eq!(snapshot.project(research, 0, 0)?.exposure, 0);
    assert_eq!(snapshot.project(research, 0, 1)?.exposure, 73);
    assert_eq!(snapshot.equipment(research, 0)?.remaining_service, service);
    assert_eq!(snapshot.stock(research, 0)?.stock, 4);
    assert_eq!(snapshot.strike(research, 0)?.probability, 0);
    snapshot.words.pop();
    assert_eq!(
        snapshot.strike(research, 0).unwrap_err().to_string(),
        "incomplete research observation"
    );
    Ok(())
}

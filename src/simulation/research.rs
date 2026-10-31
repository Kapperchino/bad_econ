pub mod capabilities;
mod kernels;
mod layout;
mod nonlinear;
mod random;

use self::{
    kernels::{ResearchStage, research},
    layout::*,
};
use super::{
    Device,
    flags::{CONTROL_WORDS, FaultFlags},
};
use crate::arithmetic::hex;
use crate::domain::{Phase, Scenario, State, research::*};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use bytemuck::{Pod, Zeroable};
use cubecl::{CubeCount, CubeDim, prelude::BufferArg};
pub use kernels::{AttemptOutcome, Participation};
use sha2::{Digest, Sha256};
use std::cell::Cell;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EquipmentObservation {
    pub remaining_service: u64,
    pub condition: u32,
    pub maintenance_consumed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabObservation {
    pub discovered: u32,
    pub mastered: u32,
    pub permitted: u32,
    pub adopted: u32,
    pub consumed: [u64; 3],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectObservation {
    pub attempts: u64,
    pub discoveries: u64,
    pub exposure: u32,
    pub outcome: AttemptOutcome,
    pub probability: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StockObservation {
    pub stock: u32,
    pub extracted: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrikeObservation {
    pub score: i32,
    pub probability: u32,
    pub participation: Participation,
}

struct SnapshotOffsets {
    projects: usize,
    machines: usize,
    stocks: usize,
    strikes: usize,
}
impl SnapshotOffsets {
    fn new(config: &ResearchConfig) -> Self {
        let projects = config.labs.len() * LabState::WORDS;
        let project_count = config
            .labs
            .iter()
            .map(|lab| lab.projects.len())
            .sum::<usize>();
        let machines = projects + project_count * ProjectState::WORDS;
        let stocks = machines + config.equipment.len() * MachineState::WORDS;
        let strikes = stocks + config.natural_stocks.len() * StockState::WORDS;
        Self {
            projects,
            machines,
            stocks,
            strikes,
        }
    }
}

impl ResearchSnapshot {
    fn record<T: Pod>(&self, offset: usize, words: usize) -> Result<T> {
        let record = self
            .words
            .get(offset..offset + words)
            .context("incomplete research observation")?;
        Ok(bytemuck::pod_read_unaligned(bytemuck::cast_slice(record)))
    }

    pub fn lab(&self, config: &ResearchConfig, index: usize) -> Result<LabObservation> {
        let _ = config.labs.get(index).context("unknown lab observation")?;
        let state: LabState = self.record(index * LabState::WORDS, LabState::WORDS)?;
        Ok(LabObservation {
            discovered: state.discovered,
            mastered: state.mastered,
            permitted: state.permitted,
            adopted: state.adopted,
            consumed: state.consumed.map(u64::from),
        })
    }

    pub fn project(
        &self,
        config: &ResearchConfig,
        lab: usize,
        project: usize,
    ) -> Result<ProjectObservation> {
        let _ = config
            .labs
            .get(lab)
            .and_then(|lab| lab.projects.get(project))
            .context("unknown project observation")?;
        let index = config.labs[..lab]
            .iter()
            .map(|lab| lab.projects.len())
            .sum::<usize>()
            + project;
        let offset = SnapshotOffsets::new(config).projects + index * ProjectState::WORDS;
        let state: ProjectState = self.record(offset, ProjectState::WORDS)?;
        let outcome = match state.outcome {
            0 => AttemptOutcome::Idle,
            1 => AttemptOutcome::Ineligible,
            2 => AttemptOutcome::MissingResources,
            3 => AttemptOutcome::NoTime,
            4 => AttemptOutcome::Failed,
            5 => AttemptOutcome::Discovered,
            _ => Err(anyhow!("unknown research attempt outcome"))?,
        };
        Ok(ProjectObservation {
            attempts: state.attempts.into(),
            discoveries: state.discoveries.into(),
            exposure: state.exposure,
            outcome,
            probability: state.probability,
        })
    }

    pub fn stock(&self, config: &ResearchConfig, index: usize) -> Result<StockObservation> {
        let _ = config
            .natural_stocks
            .get(index)
            .context("unknown stock observation")?;
        let offset = SnapshotOffsets::new(config).stocks + index * StockState::WORDS;
        let state: StockState = self.record(offset, StockState::WORDS)?;
        Ok(StockObservation {
            stock: state.stock,
            extracted: state.extracted.into(),
        })
    }

    pub fn strike(&self, config: &ResearchConfig, index: usize) -> Result<StrikeObservation> {
        let _ = config
            .strikes
            .get(index)
            .context("unknown strike observation")?;
        let offset = SnapshotOffsets::new(config).strikes + index * StrikeState::WORDS;
        let state: StrikeState = self.record(offset, StrikeState::WORDS)?;
        let participation = match state.participation {
            0 => Participation::Working,
            1 => Participation::Striking,
            _ => Err(anyhow!("unknown strike participation"))?,
        };
        Ok(StrikeObservation {
            score: state.score as i32 - 4500,
            probability: state.probability,
            participation,
        })
    }
    pub fn equipment(&self, config: &ResearchConfig, index: usize) -> Result<EquipmentObservation> {
        let _ = config
            .equipment
            .get(index)
            .context("unknown equipment observation")?;
        let offset = SnapshotOffsets::new(config).machines + index * MachineState::WORDS;
        let state: MachineState = self.record(offset, MachineState::WORDS)?;
        Ok(EquipmentObservation {
            remaining_service: state.remaining.into(),
            condition: state.condition,
            maintenance_consumed: state.consumed.into(),
        })
    }
}

struct EncodedResearch {
    configuration: Vec<u32>,
    state: Vec<u32>,
    work_items: usize,
    configuration_hash: String,
}

struct ResearchLaunch<'a> {
    main: &'a CubeTensor,
    state: &'a CubeTensor,
    output: &'a CubeTensor,
    stage: ResearchStage,
    day: u32,
    count: usize,
}

fn append<T: Pod>(words: &mut Vec<u32>, values: &[T]) -> Result<u32> {
    let offset = u32::try_from(words.len()).context("research buffer capacity exceeded")?;
    words.extend_from_slice(bytemuck::cast_slice(values));
    Ok(offset)
}

impl EncodedResearch {
    fn new(config: &ResearchConfig, scenario: &Scenario, initial: &State) -> Result<Self> {
        let config = ResearchConfig::new(config.clone(), &initial.actors)?;
        let actor_index = |identity| {
            scenario
                .index_of(identity)
                .context("unknown research actor")
        };
        let mut labs = Vec::new();
        let mut lab_states = Vec::new();
        let mut projects = Vec::new();
        let mut project_states = Vec::new();
        for lab in &config.labs {
            labs.push(LabLayout {
                owner: actor_index(lab.owner)? as u32,
                researcher: actor_index(lab.researcher)? as u32,
                first_project: projects.len() as u32,
                project_count: lab.projects.len() as u32,
                adoption: match lab.adoption {
                    AdoptionPolicy::Manual => 0,
                    AdoptionPolicy::MasterAndAdoptPermitted => 1,
                },
                consent: match lab.consent {
                    ResearchConsent::Granted => 1,
                    ResearchConsent::Withheld => 0,
                },
                production_requirements: lab.production_requirements,
                discovered: lab.discovered,
                mastered: lab.mastered,
                permitted: lab.permitted,
                adopted: lab.adopted,
            });
            lab_states.push(LabState {
                discovered: lab.discovered,
                mastered: lab.mastered,
                permitted: lab.permitted,
                adopted: lab.adopted,
                ..LabState::zeroed()
            });
            for project in &lab.projects {
                projects.push(ProjectLayout {
                    identity: project.identity.into(),
                    prerequisites: project.prerequisites,
                    outcomes: project.outcomes,
                    inputs: project.inputs,
                    labor_minutes: project.labor_minutes,
                    minimum_effort: project.minimum_effort,
                    pioneer_effort: project.pioneer_effort,
                    capacity: project.capacity,
                    relatedness: project.relatedness,
                    shape: project.shape,
                    signal: project.exposure_signal,
                });
                project_states.push(ProjectState {
                    exposure: project.exposure,
                    ..ProjectState::zeroed()
                });
            }
        }
        let machines = config
            .equipment
            .iter()
            .map(|machine| {
                let firm = actor_index(machine.firm)?;
                Ok(MachineLayout {
                    firm: firm as u32,
                    output_ticks: scenario.technique(firm).output_ticks,
                    initial_service: machine.remaining_service.into(),
                    maximum_packets: initial.actors[firm].capacity,
                    nameplate: machine.nameplate,
                    downtime: machine.downtime,
                    wear: machine.wear_per_packet,
                    maintenance_good: machine.maintenance_good as u32,
                    maintenance_ticks: machine.maintenance_ticks,
                    restoration: machine.restoration_per_tick,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let machine_states = config
            .equipment
            .iter()
            .map(|machine| MachineState {
                remaining: machine.remaining_service.into(),
                condition: machine.condition,
                ..MachineState::zeroed()
            })
            .collect::<Vec<_>>();
        let stocks = config
            .natural_stocks
            .iter()
            .map(|stock| {
                let firm = actor_index(stock.firm)?;
                Ok(StockLayout {
                    firm: firm as u32,
                    output_ticks: scenario.technique(firm).output_ticks,
                    maximum_packets: initial.actors[firm].capacity,
                    capacity: stock.capacity,
                    renewal: stock.renewal_rate,
                    extraction: stock.extraction_per_packet,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let stock_states = config
            .natural_stocks
            .iter()
            .map(|stock| StockState {
                stock: stock.stock,
                ..StockState::zeroed()
            })
            .collect::<Vec<_>>();
        let strikes = config
            .strikes
            .iter()
            .map(|strike| {
                Ok(StrikeLayout {
                    household: actor_index(strike.household)? as u32,
                    identity: strike.household.into(),
                    grievance: strike.grievance,
                    solidarity: strike.solidarity,
                    trust: strike.trust,
                    efficacy: strike.efficacy,
                    peers: strike.peers,
                    cost: strike.cost,
                    risk: strike.risk,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let strike_states = vec![StrikeState::zeroed(); strikes.len()];
        let mut configuration = vec![0; Header::WORDS];
        let mut state = Vec::new();
        let header = Header {
            actors: scenario.actors() as u32,
            lab_count: labs.len() as u32,
            project_count: projects.len() as u32,
            machine_count: machines.len() as u32,
            stock_count: stocks.len() as u32,
            strike_count: strikes.len() as u32,
            labs: append(&mut configuration, &labs)?,
            projects: append(&mut configuration, &projects)?,
            machines: append(&mut configuration, &machines)?,
            stocks: append(&mut configuration, &stocks)?,
            strikes: append(&mut configuration, &strikes)?,
            lab_state: append(&mut state, &lab_states)?,
            project_state: append(&mut state, &project_states)?,
            machine_state: append(&mut state, &machine_states)?,
            stock_state: append(&mut state, &stock_states)?,
            strike_state: append(&mut state, &strike_states)?,
            seed: scenario.config().seed.into(),
        };
        configuration[..Header::WORDS].copy_from_slice(bytemuck::cast_slice(&[header]));
        state.resize(state.len().max(1), 0);
        let work_items = [
            labs.len(),
            projects.len(),
            machines.len(),
            stocks.len(),
            strikes.len(),
            1,
        ]
        .into_iter()
        .max()
        .unwrap_or(1);
        let mut hash = Sha256::new();
        hash.update(b"bad-econ/research-configuration/ResearchFixedV1\0");
        hash.update(serde_json::to_vec(&config)?);
        for &word in configuration.iter().chain(&state) {
            hash.update(word.to_le_bytes());
        }
        Ok(Self {
            configuration,
            state,
            work_items,
            configuration_hash: hex(&hash.finalize()),
        })
    }
}

pub struct ResearchRuntime {
    configuration: CubeTensor,
    state: CubeTensor,
    audited: CubeTensor,
    table: CubeTensor,
    constants: CubeTensor,
    summary: CubeTensor,
    control: CubeTensor,
    dummy: CubeTensor,
    work_items: usize,
    state_words: usize,
    readback_bytes: Cell<u64>,
    configuration_hash: String,
}

impl ResearchRuntime {
    pub fn new(
        device: &Device,
        config: &ResearchConfig,
        scenario: &Scenario,
        initial: &State,
    ) -> Result<Self> {
        let encoded = EncodedResearch::new(config, scenario, initial)?;
        let runtime = Self {
            configuration: device.tensor(&encoded.configuration),
            state: device.tensor(&encoded.state),
            audited: device.empty(encoded.state.len()),
            table: device.tensor(&nonlinear::HAZARD_TABLE),
            constants: device.tensor(&random::SHA_CONSTANTS),
            summary: device.empty(SummaryLayout::WORDS),
            control: device.tensor(&[0; CONTROL_WORDS]),
            dummy: device.tensor(&[0; 2]),
            work_items: encoded.work_items,
            state_words: encoded.state.len(),
            readback_bytes: Cell::new(0),
            configuration_hash: encoded.configuration_hash,
        };
        runtime.save_audited(device)?;
        Ok(runtime)
    }

    pub fn restore(
        device: &Device,
        config: &ResearchConfig,
        scenario: &Scenario,
        initial: &State,
        snapshot: &ResearchSnapshot,
    ) -> Result<Self> {
        let runtime = Self::new(device, config, scenario, initial)?;
        let admitted = snapshot.profile == RESEARCH_PROFILE
            && snapshot.table_hash == RESEARCH_TABLE_HASH
            && snapshot.words.len() == runtime.state_words
            && snapshot.configuration_hash == runtime.configuration_hash;
        match admitted {
            false => Err(anyhow!(
                "research checkpoint profile, table, configuration or shape mismatch"
            )),
            true => {
                device.write(&runtime.state, &snapshot.words);
                runtime.run(
                    device,
                    &runtime.dummy,
                    &runtime.summary,
                    ResearchStage::Audit,
                    0,
                    runtime.work_items,
                )?;
                match runtime.summary(device)? == snapshot.summary {
                    true => {
                        runtime.save_audited(device)?;
                        Ok(runtime)
                    }
                    false => Err(anyhow!("research checkpoint summary mismatch")),
                }
            }
        }
    }

    fn run(
        &self,
        device: &Device,
        main: &CubeTensor,
        output: &CubeTensor,
        stage: ResearchStage,
        day: u32,
        count: usize,
    ) -> Result<()> {
        device.write(&self.control, &[0; CONTROL_WORDS]);
        self.dispatch(
            device,
            ResearchLaunch {
                main,
                state: &self.state,
                output,
                stage,
                day,
                count,
            },
        );
        self.readback_bytes
            .set(self.readback_bytes.get() + (CONTROL_WORDS * 4) as u64);
        FaultFlags::read(&device.read(&self.control)?)?.into_result()
    }

    fn dispatch(&self, device: &Device, launch: ResearchLaunch<'_>) {
        let argument = |tensor: &CubeTensor| unsafe {
            BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
        };
        let groups = (launch.count as u32).div_ceil(device.workgroup).max(1);
        let x = groups.min(u16::MAX as u32);
        research::launch(
            &device.client,
            CubeCount::Static(x, groups.div_ceil(x), 1),
            CubeDim::new_1d(device.workgroup),
            argument(launch.main),
            argument(launch.state),
            argument(&self.configuration),
            argument(&self.table),
            argument(&self.constants),
            argument(launch.output),
            argument(&self.control),
            launch.day,
            launch.stage,
        );
    }

    pub fn before_phase(
        &self,
        device: &Device,
        main: &CubeTensor,
        phase: Phase,
        day: u64,
    ) -> Result<()> {
        let day = u32::try_from(day).context("research date capacity exhausted")?;
        let stages: &[ResearchStage] = match phase {
            Phase::PhysicalClose => &[ResearchStage::Maintenance, ResearchStage::Attempt],
            _ => &[],
        };
        stages.iter().try_for_each(|&stage| {
            self.run(device, main, &self.summary, stage, day, self.work_items)
        })
    }

    pub fn phase(&self, device: &Device, main: &CubeTensor, phase: Phase, day: u64) -> Result<()> {
        let day = u32::try_from(day).context("research date capacity exhausted")?;
        let stages: &[ResearchStage] = match phase {
            Phase::Opening => &[ResearchStage::Opening],
            Phase::Observation => &[ResearchStage::Strike],
            Phase::Proposal => &[
                ResearchStage::ProductionReadiness,
                ResearchStage::MachineCapacity,
                ResearchStage::StockCapacity,
            ],
            Phase::PhysicalClose => &[ResearchStage::MachineClose, ResearchStage::StockClose],
            Phase::Audit => &[ResearchStage::Audit],
            _ => &[],
        };
        stages.iter().try_for_each(|&stage| {
            self.run(device, main, &self.summary, stage, day, self.work_items)
        })
    }

    pub fn summary(&self, device: &Device) -> Result<ResearchSummary> {
        self.run(
            device,
            &self.dummy,
            &self.summary,
            ResearchStage::Summary,
            0,
            SummaryLayout::WORDS / 2,
        )?;
        let words = device.read(&self.summary)?;
        self.readback_bytes
            .set(self.readback_bytes.get() + (SummaryLayout::WORDS * 4) as u64);
        let summary: &SummaryLayout = bytemuck::from_bytes(bytemuck::cast_slice(&words));
        Ok(ResearchSummary {
            consumed: summary.consumed.map(u64::from),
            attempts: summary.attempts.into(),
            discoveries: summary.discoveries.into(),
            remaining_service: summary.remaining_service.into(),
            natural_stock: summary.natural_stock.into(),
            striking: summary.striking.into(),
        })
    }

    pub fn snapshot(&self, device: &Device) -> Result<ResearchSnapshot> {
        self.readback_bytes
            .set(self.readback_bytes.get() + (self.state_words * 4) as u64);
        Ok(ResearchSnapshot {
            profile: RESEARCH_PROFILE.into(),
            table_hash: RESEARCH_TABLE_HASH.into(),
            configuration_hash: self.configuration_hash.clone(),
            words: device.read(&self.state)?,
            summary: self.summary(device)?,
        })
    }

    pub fn audited_snapshot(&self, device: &Device) -> Result<ResearchSnapshot> {
        device.write(&self.control, &[0; CONTROL_WORDS]);
        self.dispatch(
            device,
            ResearchLaunch {
                main: &self.dummy,
                state: &self.audited,
                output: &self.summary,
                stage: ResearchStage::Summary,
                day: 0,
                count: SummaryLayout::WORDS / 2,
            },
        );
        self.readback_bytes.set(
            self.readback_bytes.get()
                + ((CONTROL_WORDS + SummaryLayout::WORDS + self.state_words) * 4) as u64,
        );
        FaultFlags::read(&device.read(&self.control)?)?.into_result()?;
        let words = device.read(&self.summary)?;
        let summary: &SummaryLayout = bytemuck::from_bytes(bytemuck::cast_slice(&words));
        Ok(ResearchSnapshot {
            profile: RESEARCH_PROFILE.into(),
            table_hash: RESEARCH_TABLE_HASH.into(),
            configuration_hash: self.configuration_hash.clone(),
            words: device.read(&self.audited)?,
            summary: ResearchSummary {
                consumed: summary.consumed.map(u64::from),
                attempts: summary.attempts.into(),
                discoveries: summary.discoveries.into(),
                remaining_service: summary.remaining_service.into(),
                natural_stock: summary.natural_stock.into(),
                striking: summary.striking.into(),
            },
        })
    }

    pub fn readback_bytes(&self) -> u64 {
        self.readback_bytes.get()
    }

    pub fn save_audited(&self, device: &Device) -> Result<()> {
        self.run(
            device,
            &self.dummy,
            &self.audited,
            ResearchStage::Copy,
            0,
            self.state_words,
        )
    }

    pub fn restore_audited(&self, device: &Device) -> Result<()> {
        device.write(&self.control, &[0; CONTROL_WORDS]);
        self.dispatch(
            device,
            ResearchLaunch {
                main: &self.dummy,
                state: &self.audited,
                output: &self.state,
                stage: ResearchStage::Copy,
                day: 0,
                count: self.state_words,
            },
        );
        self.readback_bytes
            .set(self.readback_bytes.get() + (CONTROL_WORDS * 4) as u64);
        FaultFlags::read(&device.read(&self.control)?)?.into_result()
    }
}

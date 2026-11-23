use crate::ledger::{Bundle, Decision, MAX_LEGS, OrderedEvents};
use crate::simulation::{
    Device, Trace, decode,
    economy::parameters::SchedulerParameters,
    encode,
    scheduler::SchedulerStage,
    storage::{
        buffers::SchedulerBuffers,
        flags::{CONTROL_WORDS, ControlLayout, DependencyShape, EventStatus, SchedulerControl},
        layout::*,
    },
};
use anyhow::{Context, Result, anyhow};
use bytemuck::Zeroable;

pub(in crate::simulation) struct Execution {
    pub(in crate::simulation) decisions: Vec<Decision>,
    pub(in crate::simulation) depth: usize,
    pub(in crate::simulation) events: usize,
    pub(in crate::simulation) readback_bytes: u64,
    pub(in crate::simulation) status_start: usize,
}
pub(in crate::simulation) fn execute(
    device: &Device,
    buffers: SchedulerBuffers<'_>,
    count: usize,
    trace: Trace,
) -> Result<Execution> {
    let SchedulerBuffers {
        scratch, control, ..
    } = buffers;
    let mut result = Execution {
        decisions: Vec::new(),
        depth: 0,
        events: 0,
        readback_bytes: 0,
        status_start: 0,
    };
    match count {
        0 => (),
        _ => {
            let total = count * MAX_LEGS;
            let parameters = SchedulerParameters {
                count: count as u32,
                ..Default::default()
            };
            let launch = |entry: SchedulerStage, size, parameters: SchedulerParameters| {
                buffers.schedule(device, entry, size, parameters);
            };
            launch(
                SchedulerStage::Initialize,
                total.max(CONTROL_WORDS),
                parameters,
            );
            let mut source = 0;
            let mut width = 1;
            while width < total {
                launch(
                    SchedulerStage::Merge,
                    total,
                    SchedulerParameters {
                        width: width as u32,
                        source: source as u32,
                        ..parameters
                    },
                );
                source = total - source;
                width *= 2;
            }
            launch(
                SchedulerStage::Dependencies,
                total,
                SchedulerParameters {
                    source: source as u32,
                    ..parameters
                },
            );
            launch(SchedulerStage::CheckChains, count, parameters);
            let counters = SchedulerControl::new(&device.read_range(control, 0, CONTROL_WORDS)?)?;
            result.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
            let mut status_offset = 0;
            counters.faults.into_result()?;
            match counters.shape() {
                DependencyShape::Chains => {
                    launch(SchedulerStage::Chains, count, parameters);
                    let counters =
                        SchedulerControl::new(&device.read_range(control, 0, CONTROL_WORDS)?)?;
                    result.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
                    result.depth = counters.progress;
                    result.events = counters.completed;
                    counters.faults.into_result()?;
                }
                DependencyShape::Frontiers => {
                    let mut pending = count;
                    while pending > 0 {
                        launch(SchedulerStage::Clear, 1, parameters);
                        launch(
                            SchedulerStage::Frontier,
                            count,
                            SchedulerParameters {
                                status_offset: status_offset as u32,
                                ..parameters
                            },
                        );
                        status_offset = count - status_offset;
                        result.depth += 1;
                        let counters = SchedulerControl::new(&device.read_range(
                            control,
                            0,
                            CONTROL_WORDS,
                        )?)?;
                        result.readback_bytes += (CONTROL_WORDS * WORD_BYTES) as u64;
                        pending = counters.progress;
                        result.events = counters.completed;
                        counters.faults.into_result()?;
                        match result.depth <= count {
                            true => Ok(()),
                            false => Err(anyhow!("stalled dependency graph")),
                        }?;
                    }
                }
            }
            result.status_start = total * ScratchLayout::STATUSES + status_offset;
            match trace {
                Trace::Record => {
                    let statuses = device.read_range(
                        scratch,
                        total * ScratchLayout::STATUSES + status_offset,
                        count,
                    )?;
                    result.readback_bytes += (count * WORD_BYTES) as u64;
                    result.decisions = statuses
                        .into_iter()
                        .map(|code| EventStatus::try_from(code)?.decision())
                        .collect::<Result<Vec<_>>>()?
                        .into_iter()
                        .flatten()
                        .collect();
                    result.events = result.decisions.len();
                }
                Trace::Discard => (),
            }
        }
    }
    Ok(result)
}

pub struct LedgerResult {
    pub state: Vec<u64>,
    pub decisions: Vec<Decision>,
    pub frontier_depth: usize,
}
pub fn ledger(device: &Device, balances: &[u64], bundles: &[Bundle]) -> Result<LedgerResult> {
    OrderedEvents::new(bundles)?;
    match bundles
        .iter()
        .flat_map(Bundle::legs)
        .all(|leg| (leg.coordinate.0 as usize) < balances.len())
    {
        true => Ok(()),
        false => Err(anyhow!("invalid ledger coordinate")),
    }?;
    let state = device.tensor(bytemuck::cast_slice(&encode(balances)));
    let events = device.tensor(bytemuck::cast_slice(
        &bundles.iter().map(Bundle::encoded_legs).collect::<Vec<_>>(),
    ));
    let scratch =
        device.empty((bundles.len() * ScratchLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let links = device.empty((bundles.len() * LinkLayout::WORDS_PER_EVENT).max(MIN_BUFFER_WORDS));
    let control = device.tensor(bytemuck::cast_slice(&[ControlLayout::zeroed()]));
    let result = execute(
        device,
        SchedulerBuffers {
            state: &state,
            events: &events,
            scratch: &scratch,
            control: &control,
            links: &links,
        },
        bundles.len(),
        Trace::Record,
    )?;
    let mut recorded = result.decisions.into_iter();
    let decisions = bundles
        .iter()
        .map(|bundle| match bundle.legs().is_empty() {
            true => Ok(Decision::Accepted),
            false => recorded.next().context("missing ledger decision"),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(LedgerResult {
        state: decode(&device.read(&state)?),
        decisions,
        frontier_depth: result.depth,
    })
}

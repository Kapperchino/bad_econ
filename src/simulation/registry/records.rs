use super::*;
use crate::simulation::parallel;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub(in crate::simulation) struct PendingRecord {
    pub actor: u32,
    pub request: u32,
    pub birth: u32,
}

pub(in crate::simulation) struct PendingRecords {
    pub records: Vec<PendingRecord>,
    pub readback_bytes: u64,
}

#[derive(Clone, Copy)]
pub(in crate::simulation) struct RecordBuffers<'a> {
    pub requests: &'a CubeTensor,
    pub births: &'a CubeTensor,
    pub scratch: &'a CubeTensor,
}

impl PendingRecords {
    #[cfg(test)]
    pub fn from_dense(requests: &[u32], births: &[u32]) -> Result<Self> {
        match requests.len() == births.len() {
            true => Ok(Self {
                records: requests
                    .iter()
                    .zip(births)
                    .enumerate()
                    .filter(|(_, (request, birth))| **request != 0 || **birth != 0)
                    .map(|(actor, (&request, &birth))| PendingRecord {
                        actor: actor as u32,
                        request,
                        birth,
                    })
                    .collect(),
                readback_bytes: 0,
            }),
            false => Err(anyhow::anyhow!("registry request and birth counts differ")),
        }
    }

    pub fn read(device: &Device, buffers: RecordBuffers<'_>, output: &CubeTensor) -> Result<Self> {
        let RecordBuffers {
            requests,
            births,
            scratch,
        } = buffers;
        let actors = requests.meta.num_elements();
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        parallel::scan(
            device,
            scratch,
            actors,
            actors.div_ceil(device.workgroup as usize),
        );
        RecordCompaction::launch(
            device,
            device.count(actors),
            CubeDim::new_1d(device.workgroup),
            RecordCompactionInputLaunch {
                requests: argument(requests),
                births: argument(births),
                scratch: argument(scratch),
                output: argument(output),
            },
        );
        let count = device.read_range(output, 0, 1)?[0] as usize;
        let words = device.read_range(output, 1, count * 3)?;
        let records = words
            .chunks_exact(3)
            .map(|row| PendingRecord {
                actor: row[0],
                request: row[1],
                birth: row[2],
            })
            .collect();
        Ok(Self {
            records,
            readback_bytes: ((1 + count * 3) * size_of::<u32>()) as u64,
        })
    }
}

pub(in crate::simulation) fn upload_requests(
    device: &Device,
    records: &[PendingRecord],
    actors: usize,
) -> CubeTensor {
    let input = device.tensor(bytemuck::cast_slice(records));
    let requests = device.empty(actors);
    for stage in [RequestStage::Clear, RequestStage::Scatter] {
        let count = match stage {
            RequestStage::Clear => actors,
            RequestStage::Scatter => records.len(),
        };
        RequestExpansion::launch(
            device,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            RequestInputLaunch {
                records: input.clone().into_buffer_arg(),
                requests: requests.clone().into_buffer_arg(),
                stage,
            },
        );
    }
    requests
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RequestStage {
    Clear,
    Scatter,
}

#[derive(CubeType, CubeLaunch)]
struct RequestInput {
    records: Box<[u32]>,
    requests: Box<[u32]>,
    #[cube(comptime)]
    stage: RequestStage,
}

struct RequestExpansion;
struct ExpandRequests;

impl System for RequestExpansion {
    type Input = RequestInput;
    type Passes = gpu_passes![ExpandRequests];
}

#[cube]
impl GpuPass<RequestInput> for ExpandRequests {
    type Output = ();

    fn execute(mut input: RequestInput) {
        let id = ABSOLUTE_POS;
        match comptime!(input.stage) {
            RequestStage::Clear => {
                if id < input.requests.len() {
                    input.requests[id] = 0;
                }
            }
            RequestStage::Scatter => {
                if id < input.records.len() / 3 {
                    let actor = input.records[id * 3] as usize;
                    input.requests[actor] = input.records[id * 3 + 1];
                }
            }
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct RecordCountBuffers {
    pub births: Box<[u32]>,
    pub scratch: Box<[u32]>,
    #[cube(comptime)]
    pub scan: parallel::ScanMode,
}

#[derive(CubeType)]
pub(in crate::simulation) struct ProposedRecord {
    pub buffers: RecordCountBuffers,
    pub request: u32,
}

#[derive(CubeType, CubeLaunch)]
struct RecordCompactionInput {
    requests: Box<[u32]>,
    births: Box<[u32]>,
    scratch: Box<[u32]>,
    output: Box<[u32]>,
}

struct RecordCompaction;
pub(in crate::simulation) struct CountRecords;
struct WriteRecords;

impl System for RecordCompaction {
    type Input = RecordCompactionInput;
    type Passes = gpu_passes![WriteRecords];
}

#[cube]
impl GpuPass<ProposedRecord> for CountRecords {
    type Output = ();

    fn execute(mut input: ProposedRecord) {
        let actor = ABSOLUTE_POS;
        let count = input.buffers.births.len();
        let mut present = 0u32;
        if actor < count {
            present = u32::cast_from(input.request > 0 || input.buffers.births[actor] > 0);
        }
        let prefix = parallel::scan_prefix(present, comptime!(input.buffers.scan));
        if actor < count {
            input.buffers.scratch[actor] = prefix.inclusive;
        }
        if UNIT_POS == 0 {
            input.buffers.scratch[count + CUBE_POS] = prefix.total;
        }
    }
}

#[cube]
impl GpuPass<RecordCompactionInput> for WriteRecords {
    type Output = ();

    fn execute(mut input: RecordCompactionInput) {
        write_records(
            &input.requests,
            &input.births,
            &mut input.scratch,
            &mut input.output,
        );
    }
}

#[cube]
fn write_records(requests: &[u32], births: &[u32], scratch: &mut [u32], output: &mut [u32]) {
    let actor = ABSOLUTE_POS;
    let count = requests.len();
    if actor == 0 {
        output[0usize] = scratch[count + ceildiv(count as u32, CUBE_DIM) as usize - 1];
    }
    if actor < count && (requests[actor] > 0 || births[actor] > 0) {
        let mut slot = scratch[actor] - 1;
        if CUBE_POS > 0 {
            slot += scratch[count + CUBE_POS - 1];
        }
        let base = slot as usize * 3 + 1;
        output[base] = actor as u32;
        output[base + 1] = requests[actor];
        output[base + 2] = births[actor];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(CubeType, CubeLaunch)]
    struct RequestsInput {
        requests: Box<[u32]>,
        records: RecordCountBuffers,
    }

    struct PendingCounting;
    struct LoadRequest;

    impl System for PendingCounting {
        type Input = RequestsInput;
        type Passes = gpu_passes![LoadRequest, CountRecords];
    }

    #[cube]
    impl GpuPass<RequestsInput> for LoadRequest {
        type Output = ProposedRecord;

        fn execute(input: RequestsInput) -> ProposedRecord {
            let mut request = 0u32;
            if ABSOLUTE_POS < input.requests.len() {
                request = input.requests[ABSOLUTE_POS];
            }
            ProposedRecord {
                buffers: input.records,
                request,
            }
        }
    }

    #[test]
    fn pending_records_preserve_sparse_actor_order_and_partial_workgroups_on_gpu() -> Result<()> {
        for workgroup in [32, 64, 128, 256] {
            let device = crate::simulation::tests::device(workgroup)?;
            for scan in [parallel::ScanMode::Shared, parallel::ScanMode::new(&device)] {
                for count in [1, 31, 32, 33, 65, 257, 4097] {
                    let requests = (0..count)
                        .map(|actor| match actor % 7 {
                            1 | 3 => actor as u32 + 1,
                            _ => 0,
                        })
                        .collect::<Vec<_>>();
                    let births = (0..count)
                        .map(|actor| match actor % 7 {
                            2 | 3 => actor as u32 % 2 + 1,
                            _ => 0,
                        })
                        .collect::<Vec<_>>();
                    let expected = (0..count)
                        .filter(|&actor| requests[actor] > 0 || births[actor] > 0)
                        .map(|actor| PendingRecord {
                            actor: actor as u32,
                            request: requests[actor],
                            birth: births[actor],
                        })
                        .collect::<Vec<_>>();
                    let mut expected_requests = requests.clone();
                    expected_requests.resize(count + 3, 0);
                    let requests = device.tensor(&requests);
                    let births = device.tensor(&births);
                    let scratch = device.tensor(&vec![u32::MAX; count * 2 + 256]);
                    let output = device.tensor(&vec![u32::MAX; count * 3 + 1]);
                    PendingCounting::launch(
                        &device,
                        device.count(count),
                        CubeDim::new_1d(workgroup),
                        RequestsInputLaunch {
                            requests: requests.clone().into_buffer_arg(),
                            records: RecordCountBuffersLaunch {
                                births: births.clone().into_buffer_arg(),
                                scratch: scratch.clone().into_buffer_arg(),
                                scan,
                            },
                        },
                    );
                    let actual = PendingRecords::read(
                        &device,
                        RecordBuffers {
                            requests: &requests,
                            births: &births,
                            scratch: &scratch,
                        },
                        &output,
                    )?;
                    assert_eq!(actual.records, expected, "{workgroup} {scan:?} {count}");
                    assert_eq!(actual.readback_bytes, (1 + expected.len() * 3) as u64 * 4);
                    let expanded = upload_requests(&device, &actual.records, count + 3);
                    assert_eq!(
                        device.read(&expanded)?,
                        expected_requests,
                        "request expansion {workgroup} {scan:?} {count}"
                    );
                }
            }
        }
        Ok(())
    }
}

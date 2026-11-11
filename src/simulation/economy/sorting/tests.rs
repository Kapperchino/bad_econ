use super::*;
use crate::{
    domain::Phase,
    simulation::{
        flags::{CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, FAULT_WORD, MarketFlag},
        parameters::Discovery,
    },
};

struct SortBuffers {
    scratch: CubeTensor,
    control: CubeTensor,
    dispatches: CubeTensor,
}

impl SortBuffers {
    fn new(device: &Device, capacity: usize) -> Self {
        Self {
            scratch: device.empty(markets::scratch_words(capacity)),
            control: device.empty(CONTROL_WORDS + 3),
            dispatches: device
                .client
                .memory_dedicated_allocation(144, |_| device.empty(36)),
        }
    }

    fn reset(&self, device: &Device, case: &SortCase) -> Vec<u32> {
        let mut words = vec![u32::MAX; self.scratch.meta.shape()[0]];
        for index in 0..case.active {
            let hash = (index as u32)
                .wrapping_mul(747_796_405)
                .wrapping_add(2_891_336_453);
            words[index] = ((hash >> 13) ^ hash) % case.capacity as u32;
            words[case.capacity + index] = index as u32;
        }
        let mut control = vec![0; CONTROL_WORDS + 3];
        control[CONSTRUCTED_EVENTS_WORD] = case.active as u32;
        control[FAULT_WORD] = match case.fault {
            Fault::Clear | Fault::Pending => 0,
            Fault::Latched => 1,
        };
        device.write(&self.scratch, &words);
        device.write(&self.control, &control);
        markets::prepare_dispatch(device, &self.control, &self.dispatches, case.capacity);
        if let Fault::Pending = case.fault {
            control[FAULT_WORD] = 1;
            device.write(&self.control, &control);
        }
        words
    }
}

#[derive(Clone, Copy)]
enum Fault {
    Clear,
    Latched,
    Pending,
}

struct SortCase {
    capacity: usize,
    active: usize,
    scan: parallel::ScanMode,
    fault: Fault,
}

impl SortCase {
    fn parameters(&self) -> EconomyParameters {
        EconomyParameters {
            scan: self.scan,
            day: self.active as u32,
            settlement_rotation: self.active as u32,
            phase: Phase::Opening,
            market: MarketFlag::Labor,
            good: 0,
            discovery: Discovery::Indexed,
            offset: 0,
            count: self.capacity as u32,
            status_start: 0,
            sort_source: 0,
            sort_width: 0,
            sort_count: 0,
        }
    }

    fn check(
        &self,
        device: &Device,
        cache: &mut MarketSort,
        actual: &SortBuffers,
        expected: &SortBuffers,
    ) -> Result<()> {
        let input = actual.reset(device, self);
        expected.reset(device, self);
        let parameters = self.parameters();
        encode(
            device,
            &expected.scratch,
            &expected.control,
            &expected.dispatches,
            &parameters,
            parallel::HistogramScan::Complete,
        );
        let source = cache.run(
            device,
            &actual.scratch,
            &actual.control,
            &actual.dispatches,
            &parameters,
        )? as usize;
        assert_eq!(
            device.read(&actual.control)?,
            device.read(&expected.control)?
        );
        let actual = device.read(&actual.scratch)?;
        let expected = device.read(&expected.scratch)?;
        assert_eq!(actual[..7 * self.capacity], expected[..7 * self.capacity]);
        match self.fault {
            Fault::Clear => {
                let mut order = (0..self.active).collect::<Vec<_>>();
                order.sort_by_key(|&index| input[index]);
                let expected_keys = order.iter().map(|&index| input[index]).collect::<Vec<_>>();
                let expected_items = order.iter().map(|&index| index as u32).collect::<Vec<_>>();
                assert_eq!(actual[source..source + self.active], expected_keys);
                assert_eq!(
                    actual[source + self.capacity..source + self.capacity + self.active],
                    expected_items
                );
            }
            Fault::Latched | Fault::Pending => assert_eq!(actual, input),
        }
        Ok(())
    }
}

#[test]
fn graph_sort_preserves_first_use_replay_dynamic_counts_and_faults_on_gpu() -> Result<()> {
    for workgroup in [32, 256] {
        let device = crate::simulation::tests::device(workgroup)?;
        let capacity = 65554;
        let actual = SortBuffers::new(&device, capacity);
        let expected = SortBuffers::new(&device, capacity);
        let mut cache = MarketSort::default();
        for scan in [parallel::ScanMode::Shared, parallel::ScanMode::new(&device)] {
            for active in [65537, 0, 1, 33, 257, capacity] {
                SortCase {
                    capacity,
                    active,
                    scan,
                    fault: Fault::Clear,
                }
                .check(&device, &mut cache, &actual, &expected)?;
            }
            for fault in [Fault::Latched, Fault::Pending] {
                for active in [0, 257] {
                    SortCase {
                        capacity,
                        active,
                        scan,
                        fault,
                    }
                    .check(&device, &mut cache, &actual, &expected)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn blocked_histogram_scatter_matches_complete_prefixes_across_scan_hierarchies_on_gpu() -> Result<()>
{
    for (workgroup, capacity) in [(32, 131_090), (256, 1_048_593)] {
        let device = crate::simulation::tests::device(workgroup)?;
        let actual = SortBuffers::new(&device, capacity);
        let expected = SortBuffers::new(&device, capacity);
        let mut cache = MarketSort::default();
        for scan in [parallel::ScanMode::Shared, parallel::ScanMode::new(&device)] {
            for active in [257, capacity - 17, capacity] {
                SortCase {
                    capacity,
                    active,
                    scan,
                    fault: Fault::Clear,
                }
                .check(&device, &mut cache, &actual, &expected)?;
            }
        }
    }
    Ok(())
}

#[test]
fn graph_sort_rebuilds_after_buffer_layout_device_workgroup_thread_and_stream_changes_on_gpu()
-> Result<()> {
    let mut cache = MarketSort::default();
    for base in [Device::headless(64)?, Device::headless(128)?] {
        let actual = SortBuffers::new(&base, 8193);
        let expected = SortBuffers::new(&base, 8193);
        for workgroup in [64, 128] {
            let mut device = base.clone();
            device.workgroup = workgroup;
            for capacity in [8193, 1025, 8193] {
                SortCase {
                    capacity,
                    active: capacity - 17,
                    scan: parallel::ScanMode::new(&device),
                    fault: Fault::Clear,
                }
                .check(&device, &mut cache, &actual, &expected)?;
            }
        }
    }
    let device = Device::headless(64)?;
    let actual = SortBuffers::new(&device, 1025);
    let expected = SortBuffers::new(&device, 1025);
    let case = SortCase {
        capacity: 1025,
        active: 193,
        scan: parallel::ScanMode::new(&device),
        fault: Fault::Clear,
    };
    case.check(&device, &mut cache, &actual, &expected)?;
    device.sync()?;
    let stream = StreamId::current();
    std::thread::spawn(move || {
        stream.executes(|| case.check(&device, &mut cache, &actual, &expected))?;
        StreamId::allocate().executes(|| {
            let actual = SortBuffers::new(&device, case.capacity);
            let expected = SortBuffers::new(&device, case.capacity);
            case.check(&device, &mut cache, &actual, &expected)
        })
    })
    .join()
    .expect("sort thread completed")?;
    Ok(())
}

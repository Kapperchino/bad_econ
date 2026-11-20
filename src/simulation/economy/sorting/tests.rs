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

    fn reset_keys(&self, device: &Device, case: &SortCase, keys: SortKeys) -> Vec<u32> {
        let mut words = self.reset(device, case);
        let span = keys.span(case.capacity as u32).max(1);
        for key in &mut words[..case.active] {
            *key = keys.base() + *key % span;
        }
        if let Some(last) = words[..case.active].last_mut() {
            *last = keys.base() + span - 1;
        }
        device.write(&self.scratch, &words);
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

#[test]
fn byte_radix_matches_nibble_stability_parity_faults_and_restored_dispatches_on_gpu() -> Result<()>
{
    let device = crate::simulation::tests::device(64)?;
    let capacity = 1_010_003;
    let nibble = SortBuffers::new(&device, capacity);
    let byte = SortBuffers::new(&device, capacity);
    for keys in [
        SortKeys::Actors,
        SortKeys::Firms {
            households: 1_000_000,
        },
        SortKeys::Firms {
            households: capacity as u32 - 16_385,
        },
        SortKeys::Firms {
            households: capacity as u32 - 20_480,
        },
        SortKeys::Firms {
            households: capacity as u32 - 20_481,
        },
        SortKeys::Firms {
            households: capacity as u32 - 257,
        },
    ] {
        let mut nibble_cache = MarketSort::Unrecorded;
        let mut byte_cache = MarketSort::Unrecorded;
        for scan in [parallel::ScanMode::new(&device), parallel::ScanMode::Shared] {
            for fault in [Fault::Clear, Fault::Latched, Fault::Pending] {
                for active in [0, 1, 257, capacity - 1] {
                    let case = SortCase {
                        capacity,
                        active,
                        scan,
                        fault,
                    };
                    let input = nibble.reset_keys(&device, &case, keys);
                    byte.reset_keys(&device, &case, keys);
                    let dispatches = device.read(&byte.dispatches)?;
                    let parameters = case.parameters();
                    let nibble_source = nibble_cache.run_radix(
                        &device,
                        &nibble.scratch,
                        &nibble.control,
                        &nibble.dispatches,
                        &parameters,
                        keys,
                        Radix::Nibble,
                    )? as usize;
                    let byte_source = byte_cache.run_radix(
                        &device,
                        &byte.scratch,
                        &byte.control,
                        &byte.dispatches,
                        &parameters,
                        keys,
                        Radix::Byte,
                    )? as usize;
                    assert_eq!(device.read(&byte.dispatches)?, dispatches);
                    assert_eq!(device.read(&byte.control)?, device.read(&nibble.control)?);
                    let nibble_words = device.read(&nibble.scratch)?;
                    let byte_words = device.read(&byte.scratch)?;
                    match fault {
                        Fault::Clear => {
                            let mut order = (0..active).collect::<Vec<_>>();
                            order.sort_by_key(|&index| input[index]);
                            let expected_keys =
                                order.iter().map(|&index| input[index]).collect::<Vec<_>>();
                            let expected_items =
                                order.iter().map(|&index| index as u32).collect::<Vec<_>>();
                            for (words, source) in
                                [(&nibble_words, nibble_source), (&byte_words, byte_source)]
                            {
                                assert_eq!(words[source..source + active], expected_keys);
                                assert_eq!(
                                    words[source + capacity..source + capacity + active],
                                    expected_items
                                );
                                assert_eq!(words[words.len() - 16..], [u32::MAX; 16]);
                            }
                            if keys.span(capacity as u32) == 257 {
                                assert_ne!(nibble_source, byte_source);
                            }
                        }
                        Fault::Latched | Fault::Pending => {
                            assert_eq!(nibble_words, input);
                            assert_eq!(byte_words, input);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn actor_and_firm_sort_graphs_survive_alternating_keyspaces_on_gpu() -> Result<()> {
    let device = crate::simulation::tests::device(64)?;
    let households = 4099;
    let capacity = households + 257;
    let buffers = SortBuffers::new(&device, capacity);
    let mut cache = MarketSort::Unrecorded;
    for keys in [
        SortKeys::Actors,
        SortKeys::Firms {
            households: households as u32,
        },
        SortKeys::Actors,
        SortKeys::Firms {
            households: households as u32,
        },
    ] {
        let case = SortCase {
            capacity,
            active: 257,
            scan: parallel::ScanMode::new(&device),
            fault: Fault::Clear,
        };
        let mut input = buffers.reset(&device, &case);
        for key in &mut input[..case.active] {
            *key = households as u32 + *key % 257;
        }
        device.write(&buffers.scratch, &input);
        let source = cache.run_keys(
            &device,
            &buffers.scratch,
            &buffers.control,
            &buffers.dispatches,
            &case.parameters(),
            keys,
        )? as usize;
        let words = device.read(&buffers.scratch)?;
        let mut order = (0..case.active).collect::<Vec<_>>();
        order.sort_by_key(|&index| input[index]);
        assert_eq!(
            words[source..source + case.active],
            order.iter().map(|&index| input[index]).collect::<Vec<_>>()
        );
        assert_eq!(
            words[source + capacity..source + capacity + case.active],
            order.iter().map(|&index| index as u32).collect::<Vec<_>>()
        );
    }
    match cache {
        MarketSort::Recorded(cache) => {
            for graph in [cache.actors, cache.firms] {
                match graph {
                    SortGraph::Recorded(recorded) => assert_eq!(recorded.replays, 1),
                    SortGraph::Unrecorded => panic!("both keyspaces should remain recorded"),
                }
            }
        }
        _ => panic!("sort graphs should remain cached"),
    }
    Ok(())
}

impl SortCase {
    fn parameters(&self) -> EconomyParameters {
        EconomyParameters {
            clock: crate::simulation::parameters::MarketClock::Scalars,
            scan: self.scan,
            empty_orders: crate::simulation::parameters::EmptyOrderMode::Settle,
            employment_term_days: 31,
            day: self.active as u32,
            matching_round: 0,
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
        let reference_source =
            SortPlan::with_radix(&parameters, SortKeys::Actors, Radix::Nibble).output() as usize;
        match self.fault {
            Fault::Clear => {
                let mut order = (0..self.active).collect::<Vec<_>>();
                order.sort_by_key(|&index| input[index]);
                let expected_keys = order.iter().map(|&index| input[index]).collect::<Vec<_>>();
                let expected_items = order.iter().map(|&index| index as u32).collect::<Vec<_>>();
                assert_eq!(actual[source..source + self.active], expected_keys);
                assert_eq!(
                    expected[reference_source..reference_source + self.active],
                    expected_keys
                );
                assert_eq!(
                    actual[source + self.capacity..source + self.capacity + self.active],
                    expected_items
                );
                assert_eq!(
                    expected[reference_source + self.capacity
                        ..reference_source + self.capacity + self.active],
                    expected_items
                );
            }
            Fault::Latched | Fault::Pending => {
                assert_eq!(actual, input);
                assert_eq!(expected, input);
            }
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

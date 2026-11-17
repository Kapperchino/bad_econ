use super::*;
use crate::domain::{HouseholdSeed, Scenario, ScenarioConfig};
use crate::simulation::{Market, Simulation};
use anyhow::Result;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClaimRecord {
    resource: u32,
    priority: u32,
    buyer: u32,
    amount: u64,
}

#[cube(launch)]
fn request_totals(records: &[u32], resources: &[Atomic<u32>], count: u32) {
    let index = ABSOLUTE_POS as u32;
    if index < count {
        let claim = Claim::load(records, index);
        let base = claim.resource as usize * RESOURCE_WORDS;
        add_demand(
            resources,
            base + REQUESTED,
            base + REQUEST_OVERFLOW,
            claim.amount,
        );
    }
}

#[test]
fn concurrent_demand_totals_preserve_high_word_carry_overflow_and_zero_additions_on_gpu()
-> Result<()> {
    struct DemandCase {
        opening: u64,
        overflow: u32,
        amounts: Vec<u64>,
    }
    let cases = [
        DemandCase {
            opening: 1,
            overflow: 0,
            amounts: vec![u64::MAX],
        },
        DemandCase {
            opening: u64::MAX,
            overflow: 0,
            amounts: vec![1],
        },
        DemandCase {
            opening: u64::MAX - 17,
            overflow: 0,
            amounts: vec![0; 257],
        },
        DemandCase {
            opening: 0,
            overflow: 0,
            amounts: vec![u64::from(u32::MAX); 1025],
        },
        DemandCase {
            opening: 0,
            overflow: 0,
            amounts: vec![1u64 << 32; 1025],
        },
        DemandCase {
            opening: 19,
            overflow: 0,
            amounts: (0..1025)
                .map(|index| match index % 4 {
                    0 => u64::MAX,
                    1 => 1,
                    2 => 1u64 << 32,
                    _ => 0,
                })
                .collect(),
        },
        DemandCase {
            opening: 7,
            overflow: 1,
            amounts: vec![0; 257],
        },
    ];
    let records = cases
        .iter()
        .enumerate()
        .flat_map(|(resource, case)| {
            case.amounts.iter().flat_map(move |&amount| {
                [resource as u32, 0, 0, amount as u32, (amount >> 32) as u32]
            })
        })
        .collect::<Vec<_>>();
    let mut initial = vec![0; cases.len() * RESOURCE_WORDS];
    for (resource, case) in cases.iter().enumerate() {
        let base = resource * RESOURCE_WORDS;
        initial[base + REQUESTED] = case.opening as u32;
        initial[base + REQUESTED + 1] = (case.opening >> 32) as u32;
        initial[base + REQUEST_OVERFLOW] = case.overflow;
    }
    for workgroup in [32, 64, 256] {
        let device = crate::simulation::tests::device(workgroup)?;
        let claims = device.tensor(&records);
        let resources = device.tensor(&initial);
        let count = records.len() / FREIGHT_CLAIM_WORDS;
        request_totals::launch(
            &device.client,
            device.count(count),
            CubeDim::new_1d(workgroup),
            claims.into_buffer_arg(),
            resources.clone().into_buffer_arg(),
            count as u32,
        );
        let actual = device.read(&resources)?;
        for (resource, case) in cases.iter().enumerate() {
            let sum = u128::from(case.opening)
                + case
                    .amounts
                    .iter()
                    .map(|&amount| u128::from(amount))
                    .sum::<u128>();
            let base = resource * RESOURCE_WORDS;
            assert_eq!(
                u64::from(actual[base + REQUESTED])
                    | (u64::from(actual[base + REQUESTED + 1]) << 32),
                sum as u64,
                "workgroup {workgroup}, resource {resource}"
            );
            assert_eq!(
                actual[base + REQUEST_OVERFLOW],
                case.overflow | u32::from(sum > u128::from(u64::MAX)),
                "workgroup {workgroup}, resource {resource}"
            );
        }
    }
    Ok(())
}

#[test]
fn sorted_gpu_prefixes_match_wide_reference_with_repeated_visits_and_overflow() -> Result<()> {
    for (workgroup, sorting) in [
        (32, SortMethod::Merge),
        (64, SortMethod::Merge),
        (256, SortMethod::Merge),
        (32, SortMethod::Block2048),
        (64, SortMethod::Block2048),
        (256, SortMethod::Block2048),
    ] {
        let mut config: ScenarioConfig =
            serde_json::from_str(include_str!("../../../../../../scenarios/railway.json"))?;
        let population = config.population.as_mut().unwrap();
        let template = population.households[0].clone();
        population
            .households
            .extend((0..129).map(|index| HouseholdSeed {
                identity: 100 + index,
                ..template.clone()
            }));
        config.households = population.households.len() as u32;
        let seller = config.households;
        let scenario = Scenario::new(config)?;
        let device = crate::simulation::tests::device(workgroup)?;
        let mut simulation = Simulation::new(scenario.clone(), device.clone())?;
        let actors = simulation.config.actors;
        let regions = simulation.config.railway.region_count;
        simulation.config.railway.freight =
            Buffers::new(&device, actors as usize, regions as usize);
        simulation.config.railway.freight.sorting = sorting.for_device(&device);
        let initial_capacity = simulation.config.railway.freight.claim_capacity;
        device.write(
            &simulation.config.railway.freight.progress,
            &[0, 0, actors * 16 + 1, 0, 0],
        );
        assert!(matches!(
            prepare_claims(&device, &mut simulation.config)?,
            ClaimWork::Ready
        ));
        let grown_capacity = simulation.config.railway.freight.claim_capacity;
        assert!(grown_capacity > initial_capacity);
        device.write(
            &simulation.config.railway.freight.progress,
            &[0, 0, 1, 0, 0],
        );
        assert!(matches!(
            prepare_claims(&device, &mut simulation.config)?,
            ClaimWork::Ready
        ));
        assert_eq!(
            simulation.config.railway.freight.claim_capacity,
            grown_capacity
        );
        device.write(
            &simulation.config.railway.freight.progress,
            &[0, 0, 0, 0, 0],
        );
        assert!(matches!(
            prepare_claims(&device, &mut simulation.config)?,
            ClaimWork::Empty
        ));
        let buffers = &simulation.config.railway.freight;
        let parameters = simulation.parameters(
            Phase::ProvisionPurchase,
            Market::Input(Good::Steel),
            0,
            actors as usize,
        )?;
        let advance = |stage| {
            launch(
                &device,
                &simulation.state,
                &simulation.actors,
                &simulation.spatial,
                &simulation.config,
                &simulation.proposals,
                &simulation.scratch,
                &simulation.control,
                &parameters,
                stage,
            )
        };
        advance(Stage::Snapshot);
        let mut records = vec![
            ClaimRecord {
                resource: seller,
                priority: 0,
                buyer: 0,
                amount: u64::MAX - 7,
            },
            ClaimRecord {
                resource: seller,
                priority: 3,
                buyer: 1,
                amount: 12,
            },
            ClaimRecord {
                resource: seller,
                priority: 9,
                buyer: 2,
                amount: 1,
            },
            ClaimRecord {
                resource: actors,
                priority: 0,
                buyer: 0,
                amount: 6,
            },
            ClaimRecord {
                resource: actors,
                priority: 4,
                buyer: 3,
                amount: 7,
            },
            ClaimRecord {
                resource: actors,
                priority: 10,
                buyer: 4,
                amount: 1,
            },
            ClaimRecord {
                resource: seller + 3,
                priority: 1,
                buyer: 5,
                amount: u64::from(u32::MAX),
            },
            ClaimRecord {
                resource: seller + 3,
                priority: 2,
                buyer: 6,
                amount: 8,
            },
            ClaimRecord {
                resource: seller + 3,
                priority: 6,
                buyer: 7,
                amount: 9,
            },
            ClaimRecord {
                resource: seller + 3,
                priority: 8,
                buyer: 8,
                amount: 1,
            },
        ];
        for buyer in 9..seller {
            for _ in 0..2 {
                records.push(ClaimRecord {
                    resource: actors + 1,
                    priority: buyer * 7,
                    buyer,
                    amount: u64::from(buyer % 5 + 1),
                });
            }
        }
        let mut budgets = vec![0u64; (actors + regions) as usize];
        budgets[seller as usize] = u64::MAX;
        budgets[seller as usize + 3] = u64::from(u32::MAX) + 10;
        budgets[actors as usize] = 11;
        budgets[actors as usize + 1] = 101;
        let mut resources = vec![0; buffers.resources.meta.num_elements()];
        for (resource, &amount) in budgets.iter().enumerate() {
            resources[resource * RESOURCE_WORDS] = amount as u32;
            resources[resource * RESOURCE_WORDS + 1] = (amount >> 32) as u32;
        }
        let mut expected = records.clone();
        expected.sort_by_key(|claim| (claim.resource, claim.priority));
        let mut sums = vec![0u128; budgets.len()];
        let mut denied = vec![0; actors as usize];
        for claim in &expected {
            sums[claim.resource as usize] += u128::from(claim.amount);
            denied[claim.buyer as usize] |= u32::from(
                sums[claim.resource as usize] > u128::from(budgets[claim.resource as usize]),
            );
        }
        records.reverse();
        records.rotate_left(17);
        let mut words = vec![0; offsets_start(buffers.claim_capacity) as usize + budgets.len() + 1];
        for (index, claim) in records.iter().enumerate() {
            words[index * FREIGHT_CLAIM_WORDS..(index + 1) * FREIGHT_CLAIM_WORDS].copy_from_slice(
                &[
                    claim.resource,
                    claim.priority,
                    claim.buyer,
                    claim.amount as u32,
                    (claim.amount >> 32) as u32,
                ],
            );
            words[buffers.claim_capacity * FREIGHT_CLAIM_WORDS + index] = index as u32;
        }
        device.write(&buffers.claims, &words);
        device.write(&buffers.resources, &resources);
        device.write(
            &buffers.accounts,
            &vec![0; buffers.accounts.meta.num_elements()],
        );
        device.write(&buffers.progress, &[0, 0, records.len() as u32, 0, 0]);
        request_totals::launch(
            &device.client,
            device.count(records.len()),
            CubeDim::new_1d(workgroup),
            buffers.claims.clone().into_buffer_arg(),
            buffers.resources.clone().into_buffer_arg(),
            records.len() as u32,
        );
        for stage in [Stage::SortClaims, Stage::ClaimOffsets, Stage::Resolve] {
            advance(stage);
        }
        let actual = device.read(&buffers.claims)?;
        let source = settlement_source(buffers) as usize;
        let sorted = actual[source..source + records.len()]
            .iter()
            .map(|&index| records[index as usize].clone())
            .collect::<Vec<_>>();
        assert_eq!(sorted, expected);
        let offsets = &actual[offsets_start(buffers.claim_capacity) as usize..];
        for resource in 0..budgets.len() {
            assert_eq!(
                offsets[resource] as usize,
                expected.partition_point(|claim| (claim.resource as usize) < resource)
            );
        }
        assert_eq!(offsets[budgets.len()] as usize, records.len());
        let accounts = device.read(&buffers.accounts)?;
        denied.resize(accounts.len() / ACCOUNT_WORDS, 0);
        assert_eq!(
            accounts
                .chunks_exact(ACCOUNT_WORDS)
                .map(|account| account[DENIED])
                .collect::<Vec<_>>(),
            denied
        );
        let actual_resources = device.read(&buffers.resources)?;
        for (resource, words) in actual_resources.chunks_exact(RESOURCE_WORDS).enumerate() {
            let budget = budgets.get(resource).copied().unwrap_or(0);
            let sum = sums.get(resource).copied().unwrap_or(0);
            assert_eq!(u64::from(words[0]) | (u64::from(words[1]) << 32), budget);
            assert_eq!(
                words[REQUEST_OVERFLOW],
                u32::from(sum > u128::from(u64::MAX))
            );
            assert_eq!(
                u64::from(words[REQUESTED]) | (u64::from(words[REQUESTED + 1]) << 32),
                sum as u64
            );
        }
    }
    Ok(())
}

struct ClaimSortProbe {
    buffers: Buffers,
    control: CubeTensor,
    initial: Vec<u32>,
    count: usize,
}

impl ClaimSortProbe {
    fn new(device: &Device, capacity: usize, count: usize, sorting: SortMethod) -> Self {
        let mut buffers = Buffers::new(device, 0, 0);
        buffers.sorting = sorting.for_device(device);
        buffers.claim_capacity = capacity;
        let mut initial = vec![0xa5a5a5a5; capacity * 7 + 3];
        for index in 0..count {
            let group = (index / 3) as u32;
            let resource = match group % 11 {
                0 => u32::MAX,
                _ => group.wrapping_mul(97) % 257,
            };
            let priority = match group % 13 {
                0 => u32::MAX,
                _ => group.wrapping_mul(31) % 4096,
            };
            initial[index * FREIGHT_CLAIM_WORDS..(index + 1) * FREIGHT_CLAIM_WORDS]
                .copy_from_slice(&[
                    resource,
                    priority,
                    index as u32,
                    index as u32 ^ 17,
                    u32::MAX,
                ]);
            initial[capacity * FREIGHT_CLAIM_WORDS + index] = index as u32;
        }
        buffers.claims = device.tensor(&initial);
        Self {
            buffers,
            control: device.tensor(&vec![0; CONTROL_WORDS + 7]),
            initial,
            count,
        }
    }

    fn reset(&self, device: &Device) {
        device.write(&self.buffers.claims, &self.initial);
        device.write(&self.buffers.progress, &[0, 0, self.count as u32, 0, 0, 0]);
        device.write(&self.control, &vec![0; CONTROL_WORDS + 7]);
    }

    fn sort(&self, device: &Device, execution: Execution) {
        sort_buffered(device, &self.buffers, &self.control, execution);
    }

    fn outcome(&self, device: &Device) -> Result<Vec<u32>> {
        let actual = device.read(&self.buffers.claims)?;
        let capacity = self.buffers.claim_capacity;
        assert!(
            actual[..capacity * FREIGHT_CLAIM_WORDS]
                == self.initial[..capacity * FREIGHT_CLAIM_WORDS]
        );
        for start in [
            capacity * FREIGHT_CLAIM_WORDS,
            capacity * (FREIGHT_CLAIM_WORDS + 1),
        ] {
            assert!(
                actual[start + self.count..start + capacity]
                    == self.initial[start + self.count..start + capacity]
            );
        }
        assert_eq!(actual[capacity * 7..], self.initial[capacity * 7..]);
        let source = settlement_source(&self.buffers) as usize;
        Ok(actual[source..source + self.count].to_vec())
    }
}

#[test]
fn block_claim_sort_preserves_stable_indices_partial_tiles_and_output_parity_on_gpu() -> Result<()>
{
    let device = crate::simulation::tests::device(64)?;
    for capacity in [0usize, 1, 31, 2047, 2048, 2049, 5000, 8193] {
        for count in [0, capacity.min(1), capacity / 2, capacity] {
            let merge = ClaimSortProbe::new(&device, capacity, count, SortMethod::Merge);
            merge.reset(&device);
            merge.sort(&device, Execution::Immediate);
            let expected = merge.outcome(&device)?;
            let blocked = ClaimSortProbe::new(&device, capacity, count, SortMethod::Block2048);
            blocked.reset(&device);
            blocked.sort(&device, Execution::Immediate);
            let actual = blocked.outcome(&device)?;
            assert_eq!(
                actual, expected,
                "capacity {capacity}, count {count}, sorting {:?}",
                blocked.buffers.sorting,
            );
        }
    }
    Ok(())
}

#[test]
fn empty_claim_sort_skips_dispatches_and_small_blocks_keep_their_dispatch_on_gpu() -> Result<()> {
    let device = crate::simulation::tests::device(64)?;
    for sorting in [SortMethod::Merge, SortMethod::Block2048] {
        let probe = ClaimSortProbe::new(&device, 0, 0, sorting);
        let dispatches = vec![0xa5a5a5a5; FREIGHT_DISPATCH_WORDS];
        for execution in [Execution::Immediate, Execution::Initial, Execution::Retry] {
            probe.reset(&device);
            device.write(&probe.buffers.dispatches, &dispatches);
            probe.sort(&device, execution);
            assert_eq!(device.read(&probe.buffers.dispatches)?, dispatches);
            assert_eq!(device.read(&probe.buffers.claims)?, probe.initial);
            assert_eq!(device.read(&probe.buffers.progress)?, [0; 6]);
        }
    }
    for capacity in [1, 128, SORT_TILE, SORT_TILE + 1] {
        let probe = ClaimSortProbe::new(&device, capacity, capacity, SortMethod::Block2048);
        probe.reset(&device);
        probe.sort(&device, Execution::Immediate);
        let workgroup = match probe.buffers.sorting {
            SortMethod::Merge => device.workgroup as usize,
            method => match capacity <= method.tile() {
                true => method.tile(),
                false => device.workgroup as usize,
            },
        };
        let dispatches = device.read(&probe.buffers.dispatches)?;
        assert_eq!(
            dispatches[SORT_DISPATCH] as usize,
            capacity.div_ceil(workgroup)
        );
        let expected = ClaimSortProbe::new(&device, capacity, capacity, SortMethod::Merge);
        expected.reset(&device);
        expected.sort(&device, Execution::Immediate);
        assert_eq!(probe.outcome(&device)?, expected.outcome(&device)?);
    }
    Ok(())
}

#[test]
fn block_claim_sort_preserves_disabled_retries_and_prior_gpu_faults() -> Result<()> {
    let device = crate::simulation::tests::device(64)?;
    for execution in [Execution::Initial, Execution::Retry] {
        for faults in [
            FaultFlags::empty(),
            FaultFlags::ARITHMETIC_OVERFLOW,
            FaultFlags::CLAIM_CAPACITY,
        ] {
            let probe = ClaimSortProbe::new(&device, 5000, 128, SortMethod::Block2048);
            probe.reset(&device);
            let mut control = vec![0; CONTROL_WORDS + 7];
            control[crate::simulation::flags::FAULT_WORD] = faults.bits();
            device.write(&probe.control, &control);
            probe.sort(&device, execution);
            match (execution, faults.is_empty()) {
                (Execution::Initial, true) => {
                    let merge = ClaimSortProbe::new(&device, 5000, 128, SortMethod::Merge);
                    merge.reset(&device);
                    merge.sort(&device, execution);
                    assert_eq!(probe.outcome(&device)?, merge.outcome(&device)?);
                }
                _ => assert_eq!(device.read(&probe.buffers.claims)?, probe.initial),
            }
            assert_eq!(device.read(&probe.control)?, control);
        }
    }
    Ok(())
}

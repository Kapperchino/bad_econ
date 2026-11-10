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
fn sorted_gpu_prefixes_match_wide_reference_with_repeated_visits_and_overflow() -> Result<()> {
    for workgroup in [32, 64, 256] {
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
        let initial_capacity = simulation.config.railway.freight.claim_capacity;
        device.write(
            &simulation.config.railway.freight.progress,
            &[0, 0, initial_capacity as u32 + 1, 0, 0],
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
        let mut resources = vec![0; budgets.len() * RESOURCE_WORDS];
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
        device.write(&buffers.accounts, &vec![0; actors as usize * ACCOUNT_WORDS]);
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
        let source = sorted_source(buffers.claim_capacity) as usize;
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
        assert_eq!(
            accounts
                .chunks_exact(ACCOUNT_WORDS)
                .map(|account| account[DENIED])
                .collect::<Vec<_>>(),
            denied
        );
        let actual_resources = device.read(&buffers.resources)?;
        for (resource, words) in actual_resources.chunks_exact(RESOURCE_WORDS).enumerate() {
            assert_eq!(
                u64::from(words[0]) | (u64::from(words[1]) << 32),
                budgets[resource]
            );
            assert_eq!(
                words[REQUEST_OVERFLOW],
                u32::from(sums[resource] > u128::from(u64::MAX))
            );
            assert_eq!(
                u64::from(words[REQUESTED]) | (u64::from(words[REQUESTED + 1]) << 32),
                sums[resource] as u64
            );
        }
    }
    Ok(())
}

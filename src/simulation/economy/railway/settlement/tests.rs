use super::*;
use crate::domain::Scenario;
use crate::simulation::Simulation;
use anyhow::Result;

#[derive(Clone, Copy)]
enum Seller {
    Firm(Good),
    Household,
}

#[derive(Clone, Copy)]
enum FreightLoad {
    Available,
    Sparse,
    Constrained,
    Hundred,
    SparseConstrained,
    CashConstrained,
    Cascading,
    Empty,
}

impl FreightLoad {
    fn accepts(self, buyer: usize) -> bool {
        match self {
            Self::Available
            | Self::Constrained
            | Self::Hundred
            | Self::CashConstrained
            | Self::Cascading => true,
            Self::Sparse | Self::SparseConstrained => buyer % 3 == 0,
            Self::Empty => false,
        }
    }

    fn capacity(self) -> u64 {
        match self {
            Self::Constrained | Self::SparseConstrained => 7,
            Self::Hundred => 100,
            Self::Cascading => 5,
            _ => 1000,
        }
    }

    fn budget(self) -> u64 {
        match self {
            Self::CashConstrained => 49,
            Self::Cascading => 7,
            _ => 10_000,
        }
    }

    fn sender(self, buyer: usize, first: usize) -> usize {
        first
            + match self {
                Self::Cascading => match buyer / 5 {
                    0 => 0,
                    1 => 3,
                    _ => 4,
                },
                _ => 0,
            }
    }
}

fn freight(source: Seller, load: FreightLoad, workgroup: u32) -> Result<()> {
    use crate::domain::{HouseholdSeed, Position, ScenarioConfig};
    use crate::simulation::{Market, decode, encode};
    let mut config: ScenarioConfig =
        serde_json::from_str(include_str!("../../../../../scenarios/railway.json"))?;
    config.railway.units_per_level = load.capacity() as u32;
    let population = config.population.as_mut().unwrap();
    let template = population.households[0].clone();
    population
        .households
        .extend((0..381).map(|index| HouseholdSeed {
            identity: 100 + index,
            ..template.clone()
        }));
    if let FreightLoad::Hundred = load {
        population.households.truncate(101);
    }
    if let FreightLoad::Cascading = load {
        population.households.truncate(15);
    }
    for household in &mut population.households {
        household.position = Position { x: 900, y: 0 };
    }
    config.households = population.households.len() as u32;
    let buyers = config.households as usize;
    let population = config.population.as_mut().unwrap();
    let good = match source {
        Seller::Firm(good) => {
            population.firms[0].good = good;
            good
        }
        Seller::Household => {
            population.households[buyers - 1].position = population.firms[0].position;
            Good::Bread
        }
    };
    let seller = match source {
        Seller::Firm(_) => buyers,
        Seller::Household => buyers - 1,
    };
    let shipments = buyers.min(seller);
    let scenario = Scenario::new(config)?;
    let device = crate::simulation::tests::device(workgroup)?;
    let simulation = Simulation::new(scenario.clone(), device.clone())?;
    let mut state = scenario.initial()?;
    state.set(Field::Cash, seller, load.budget());
    if let FreightLoad::Cascading = load {
        state.set(Field::Cash, seller + 3, load.budget());
        state.set(Field::Cash, seller + 4, load.budget());
    }
    let count = state.actors.len();
    for firm in [buyers + 1, buyers + 2] {
        state.set(Field::RailBuilt, firm, 1);
        state.set(Field::RailOperating, firm, 1);
        state.set(Field::Railway, firm, load.capacity());
    }
    for region in 0..scenario.rail_geography().regions.len() {
        for (field, amount) in [
            (RailCapacityField::Level, 1),
            (RailCapacityField::Available, load.capacity()),
        ] {
            let coordinate = scenario
                .rail_geography()
                .coordinate(&scenario, &state, field, region);
            state.values[coordinate] = amount;
        }
    }
    let mut parameters =
        simulation.parameters(Phase::ProvisionPurchase, Market::Input(good), 0, count)?;
    parameters.discovery = Discovery::Indexed;
    parameters.sort_count = shipments as u32;
    let mut proposals = vec![NO_ACTOR; count * ProposalLayout::WORDS];
    let mut scratch = vec![0; markets::scratch_words(count)];
    let mut control = vec![0; CONTROL_WORDS + 4];
    control[CONSTRUCTED_EVENTS_WORD] = shipments as u32;
    let accepted = (0..shipments)
        .filter(|&buyer| load.accepts(buyer))
        .collect::<Vec<_>>();
    let delivered_buyers = match load {
        FreightLoad::Cascading => vec![0, 5],
        _ => accepted
            .iter()
            .copied()
            .take((load.capacity() as usize).min(load.budget() as usize / 7))
            .collect(),
    };
    let delivered = delivered_buyers.len();
    for buyer in 0..shipments {
        let delivery = state.shipment_coordinate(
            good,
            (parameters.day as usize + 9) % scenario.ring(),
            scenario.ring(),
            buyer,
        );
        state.values[delivery.0 as usize] = u64::from(load.accepts(buyer));
        let sender = load.sender(buyer, seller);
        proposals[buyer * ProposalLayout::WORDS + ProposalLayout::ACTOR] = sender as u32;
        proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW] = 1;
        proposals[buyer * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH] = 0;
        scratch[buyer] = sender as u32;
        scratch[count + buyer] = buyer as u32;
        scratch[4 * count + buyer] = match load.accepts(buyer) {
            true => EventStatus::Accepted as u32,
            false => EventStatus::Rejected as u32,
        };
    }
    device.write(
        &simulation.state,
        bytemuck::cast_slice(&encode(&state.values)),
    );
    device.write(&simulation.proposals, &proposals);
    device.write(&simulation.scratch, &scratch);
    device.write(&simulation.control, &control);
    for stage in [RailStage::Access, RailStage::Routes, RailStage::Quotes] {
        simulation.railways(stage);
    }
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
        );
    };
    advance(Stage::Snapshot);
    advance(Stage::Initialize);
    let buffers = &simulation.config.railway.freight;
    assert_eq!(
        device.read(&buffers.progress)?[PENDING] as usize,
        accepted.len()
    );
    let mut pending = 1;
    let mut waves = 0;
    while pending > 0 {
        advance(Stage::Queue);
        advance(Stage::Reset);
        let bookings = device.read(&buffers.bookings)?;
        let resources = device.read(&buffers.resources)?;
        advance(Stage::Propose);
        let expected_bookings = device.read(&buffers.bookings)?;
        let expected_resources = device.read(&buffers.resources)?;
        let mut fresh = bookings;
        for booking in fresh.chunks_exact_mut(FREIGHT_BOOKING_WORDS) {
            if booking[0] == BookingState::Retrying as u32 {
                booking[0] = BookingState::Pending as u32;
            }
        }
        device.write(&buffers.bookings, &fresh);
        device.write(&buffers.resources, &resources);
        advance(Stage::Propose);
        assert_eq!(device.read(&buffers.bookings)?, expected_bookings);
        assert_eq!(device.read(&buffers.resources)?, expected_resources);
        for stage in [
            Stage::CountClaims,
            Stage::EmitClaims,
            Stage::SortClaims,
            Stage::ClaimOffsets,
            Stage::Resolve,
        ] {
            advance(stage);
        }
        assert_eq!(device.read(&buffers.resources)?, expected_resources);
        let counts = device.read(&buffers.progress)?;
        let claims = counts[CLAIM_COUNT];
        assert_eq!(counts[CLAIM_CURSOR], claims);
        match load {
            FreightLoad::Available | FreightLoad::Sparse | FreightLoad::Empty => {
                assert_eq!(claims, 0)
            }
            _ => (),
        }
        advance(match waves {
            0 => Stage::Commit,
            _ => Stage::CommitFinal,
        });
        let committed = device.read(&buffers.resources)?;
        for resource in committed.chunks_exact(RESOURCE_WORDS) {
            let ticks =
                |offset| u64::from(resource[offset]) | (u64::from(resource[offset + 1]) << 32);
            assert!(ticks(CONSUMED) <= ticks(0));
        }
        advance(Stage::Release);
        pending = progress(&device, &simulation.config, &simulation.control)?.pending;
        let bookings = device.read(&buffers.bookings)?;
        if waves == 0 {
            let first_wave = match load {
                FreightLoad::Cascading => 1,
                _ => delivered,
            };
            assert_eq!(pending as usize, accepted.len() - first_wave);
        }
        let mut queued = device.read_range(&buffers.queue, count, pending as usize)?;
        queued.sort_unstable();
        let unresolved = (0..shipments)
            .filter(|&buyer| {
                bookings[buyer * FREIGHT_BOOKING_WORDS] == BookingState::Retrying as u32
            })
            .map(|buyer| buyer as u32)
            .collect::<Vec<_>>();
        assert_eq!(queued, unresolved);
        waves += 1;
        assert!(waves <= 2);
    }
    let settled = device.read(&simulation.state)?;
    let accounts = device.read(&buffers.accounts)?;
    for _ in 0..8 {
        for stage in [
            Stage::Queue,
            Stage::Reset,
            Stage::Propose,
            Stage::CountClaims,
            Stage::EmitClaims,
            Stage::SortClaims,
            Stage::ClaimOffsets,
            Stage::Resolve,
            Stage::Commit,
            Stage::Release,
        ] {
            advance(stage);
        }
    }
    assert_eq!(device.read(&simulation.state)?, settled);
    assert_eq!(device.read(&buffers.accounts)?, accounts);
    assert_eq!(
        progress(&device, &simulation.config, &simulation.control)?.pending,
        0
    );
    advance(Stage::Accounts);
    match load {
        FreightLoad::Constrained
        | FreightLoad::Hundred
        | FreightLoad::SparseConstrained
        | FreightLoad::CashConstrained
        | FreightLoad::Cascading => assert_eq!(waves, 2),
        _ => assert_eq!(waves, 1),
    }
    assert_eq!(
        device.read(&simulation.control)?[crate::simulation::flags::FAULT_WORD],
        0
    );
    let mut actual = state.clone();
    actual.values = decode(&device.read(&simulation.state)?);
    assert_eq!(
        actual.get(Field::Cash, seller),
        state.get(Field::Cash, seller)
            - delivered_buyers
                .iter()
                .filter(|&&buyer| load.sender(buyer, seller) == seller)
                .count() as u64
                * 7
    );
    assert_eq!(
        actual.get(Field::RailRevenue, buyers + 1),
        delivered as u64 * 2
    );
    assert_eq!(
        actual.get(Field::RailRevenue, buyers + 2),
        delivered as u64 * 5
    );
    assert_eq!(
        actual.get(Field::Railway, buyers + 2),
        load.capacity() - delivered as u64
    );
    for buyer in 0..shipments {
        let delivery = actual.shipment_coordinate(
            good,
            (parameters.day as usize + 1) % scenario.ring(),
            scenario.ring(),
            buyer,
        );
        assert_eq!(
            actual.values[delivery.0 as usize],
            u64::from(delivered_buyers.contains(&buyer))
        );
        let animal = actual.shipment_coordinate(
            good,
            (parameters.day as usize + 9) % scenario.ring(),
            scenario.ring(),
            buyer,
        );
        assert_eq!(
            actual.values[animal.0 as usize],
            u64::from(load.accepts(buyer) && !delivered_buyers.contains(&buyer))
        );
    }
    Ok(())
}

#[test]
fn shared_freight_settles_multiple_workgroups_in_one_gpu_wave() -> Result<()> {
    for source in [
        Seller::Firm(Good::Bread),
        Seller::Firm(Good::Steel),
        Seller::Household,
    ] {
        freight(source, FreightLoad::Available, 64)?;
    }
    Ok(())
}

#[test]
fn pending_freight_preserves_sparse_priorities_and_cached_route_retries_on_gpu() -> Result<()> {
    for workgroup in [32, 256] {
        for load in [
            FreightLoad::Sparse,
            FreightLoad::Constrained,
            FreightLoad::SparseConstrained,
            FreightLoad::CashConstrained,
            FreightLoad::Empty,
        ] {
            for source in [
                Seller::Firm(Good::Bread),
                Seller::Firm(Good::Steel),
                Seller::Household,
            ] {
                freight(source, load, workgroup)?;
            }
        }
    }
    Ok(())
}

#[test]
fn one_hundred_of_one_hundred_one_shipments_book_in_the_first_gpu_wave() -> Result<()> {
    for workgroup in [32, 64, 256] {
        freight(Seller::Firm(Good::Bread), FreightLoad::Hundred, workgroup)?;
    }
    Ok(())
}

#[test]
fn cross_resource_contention_stops_after_one_retry_on_gpu() -> Result<()> {
    for workgroup in [32, 256] {
        freight(Seller::Firm(Good::Bread), FreightLoad::Cascading, workgroup)?;
    }
    Ok(())
}

#[test]
fn market_claim_preparation_reads_gpu_searches_and_propagates_prior_faults() -> Result<()> {
    let scenario = Scenario::new(serde_json::from_str(include_str!(
        "../../../../../scenarios/railway.json"
    ))?)?;
    let device = crate::simulation::tests::device(64)?;
    let mut simulation = Simulation::new(scenario, device.clone())?;
    let mut control = vec![0; CONTROL_WORDS + 4];
    control[markets::SEARCHES_WORD] = 17;
    device.write(&simulation.control, &control);
    let preparation = prepare_market_claims(&device, &mut simulation.config, &simulation.control)?;
    assert!(matches!(preparation.claims, ClaimWork::Empty));
    assert_eq!(preparation.searches, 17);
    device.write(
        &simulation.config.railway.freight.progress,
        &[0, 0, 1, 0, 0],
    );
    control[markets::SEARCHES_WORD] = 0;
    device.write(&simulation.control, &control);
    let preparation = prepare_market_claims(&device, &mut simulation.config, &simulation.control)?;
    assert!(matches!(preparation.claims, ClaimWork::Ready));
    assert_eq!(preparation.searches, 0);
    control[crate::simulation::flags::FAULT_WORD] = FaultFlags::ARITHMETIC_OVERFLOW.bits();
    device.write(&simulation.control, &control);
    let error = prepare_market_claims(&device, &mut simulation.config, &simulation.control)
        .err()
        .expect("the prior GPU fault must propagate");
    assert!(error.to_string().contains("ARITHMETIC_OVERFLOW"));
    Ok(())
}

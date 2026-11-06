pub(super) mod management;
mod opportunities;
pub(super) mod ownership;

use super::{
    Device,
    layout::ActorLayout,
    pass::GpuPass,
    registry::records::{
        CountRecords, ProposedRecord, RecordBuffers, RecordCountBuffers, RecordCountBuffersLaunch,
    },
    system::{System, gpu_passes},
    words::*,
};
use crate::domain::{
    Field, Good,
    enterprise::{EnterpriseState, EntrepreneurialInterest},
};
use crate::simulation::configuration::{Configuration, EconomyConfig, Feature};
use burn_wgpu::CubeTensor;
use cubecl::{CubeDim, prelude::*};

#[derive(CubeType, CubeLaunch)]
struct ProposalInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    spatial: Box<[u32]>,
    output: Box<[u32]>,
    records: RecordCountBuffers,
    day: u32,
}

struct EnterpriseProposals;
struct ProposeEnterprises;

impl System for EnterpriseProposals {
    type Input = ProposalInput;
    type Passes = gpu_passes![ProposeEnterprises, CountRecords];
}

#[cube]
impl GpuPass<ProposalInput> for ProposeEnterprises {
    type Output = ProposedRecord;

    fn execute(mut input: ProposalInput) -> ProposedRecord {
        let request = requests(
            &mut input.state,
            &input.actors,
            &input.config,
            &input.spatial,
            &mut input.output,
            input.day,
        );
        ProposedRecord {
            buffers: input.records,
            request,
        }
    }
}

#[allow(
    clippy::manual_is_multiple_of,
    reason = "CubeCL lowers explicit integer remainder"
)]
#[cube]
fn requests(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    output: &mut [u32],
    day: u32,
) -> u32 {
    let actor = ABSOLUTE_POS as u32;
    let mut action = 0u32;
    if actor < config.actors {
        let count = config.actors;
        if comptime!(config.home_mode == Feature::Enabled)
            && actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + actor as usize] == 0
        {
            crate::simulation::economy::enterprises::mobility(state, actors, config, actor, day);
        }
        let target = load(
            state,
            comptime!(Field::InvestmentGood as u32) * count + actor,
        );
        let tools = load(state, comptime!(Field::Tools as u32) * count + actor);
        let wood = load(state, comptime!(Field::Wood as u32) * count + actor);
        let cash = load(state, comptime!(Field::Cash as u32) * count + actor);
        let arrears = load(state, comptime!(Field::RentArrears as u32) * count + actor);
        let funded = !lt(cash, ticks(10_000, 0))
            && (comptime!(config.property_mode == Feature::Enabled)
                || (!lt(tools, ticks(2, 0)) && !lt(wood, ticks(5, 0))))
            && equal(arrears, zero());
        let household =
            actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + actor as usize] == 0;
        let invested = load(
            state,
            comptime!(Field::InvestmentFirm as u32) * count + actor,
        );
        let living = load(
            state,
            comptime!(Field::LivingMembers as u32) * count + actor,
        );
        let eligible = comptime!(config.home_mode == Feature::Enabled)
            && household
            && actors[ActorLayout::ENTREPRENEURIAL_INTEREST * (actors.len() / ActorLayout::WORDS)
                + actor as usize]
                == comptime!(EntrepreneurialInterest::Interested as u32)
            && !equal(living, zero())
            && equal(invested, zero())
            && equal(arrears, zero())
            && !lt(cash, ticks(10_000, 0));
        let mut proposal = 0;
        let review = !equal(target, zero()) || (day + actor) % 30 == 0;
        if eligible && review {
            proposal = opportunities::discover(state, actors, config, spatial, actor, day);
            let source = select(proposal >> 24 == 8, (proposal & 0x00ffffff) - 1, actor);
            if !crate::simulation::economy::property::authorized(state, config, source, actor) {
                proposal = 0;
            }
            action = select(funded, proposal, 0);
        }
        store(
            state,
            comptime!(Field::InvestmentGood as u32) * count + actor,
            ticks(proposal, 0),
        );
        let split = load(state, comptime!(Field::SplitPerson as u32) * count + actor);
        if !equal(split, zero()) && !lt(cash, ticks(5000, 0)) {
            action = 0x01000001u32;
        }
        let moving = load(state, comptime!(Field::MoveDue as u32) * count + actor).extract(0usize);
        let destination =
            load(state, comptime!(Field::MoveTarget as u32) * count + actor).extract(0usize);
        if destination > 0 && moving == 0 && !lt(cash, ticks(100, 0)) {
            action = 0x02000000u32 + destination;
        }
        if moving > 0 && moving <= day {
            action = 0x03000000u32;
        }
        let kind = actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + actor as usize];
        let status = load(
            state,
            comptime!(Field::EnterpriseStatus as u32) * count + actor,
        )
        .extract(0usize);
        let living = load(
            state,
            comptime!(Field::LivingMembers as u32) * count + actor,
        )
        .extract(0usize);
        if kind == 0 && living == 0 && (action <= Good::COUNT as u32 || action >> 24 >= 8) {
            action = 0;
        }
        let estate = load(
            state,
            comptime!(Field::EstateSettled as u32) * count + actor,
        )
        .extract(0usize);
        if kind > 0
            && status == comptime!(EnterpriseState::Liquidating as u32)
            && (day + actor) % 30 == 0
        {
            action = 0x04000000u32;
        }
        if kind == 0 && living == 0 && (estate == 0 || (estate == 3 && (day + actor) % 30 == 0)) {
            action = 0x05000000u32;
        }
        let pending = load(
            state,
            comptime!(Field::PendingCapacity as u32) * count + actor,
        )
        .extract(0usize);
        let construction = load(
            state,
            comptime!(Field::ConstructionRemaining as u32) * count + actor,
        )
        .extract(0usize);
        if kind > 0
            && status == comptime!(EnterpriseState::Operating as u32)
            && pending > 0
            && construction == 0
        {
            action = 0x06000000u32 + pending;
        }
        let invested = load(
            state,
            comptime!(Field::EquityInvested as u32) * count + actor,
        );
        if action == 0
            && kind == 0
            && living > 0
            && equal(invested, zero())
            && !lt(cash, ticks(20_000, 0))
            && day % 30 == 0
        {
            for sample in 0..16u32 {
                let candidate = (actor + day * 17 + sample * 31) % count;
                let candidate_kind = actors
                    [ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + candidate as usize];
                let status = load(
                    state,
                    comptime!(Field::EnterpriseStatus as u32) * count + candidate,
                )
                .extract(0usize);
                let sales = load(state, comptime!(Field::Sales as u32) * count + candidate);
                let demand = load(state, comptime!(Field::Demand as u32) * count + candidate);
                let operating = load(
                    state,
                    comptime!(Field::OperatingReserve as u32) * count + candidate,
                );
                let funds = load(state, comptime!(Field::Cash as u32) * count + candidate);
                let price =
                    load(state, comptime!(Field::Price as u32) * count + candidate).extract(0usize);
                let wage =
                    load(state, comptime!(Field::Wage as u32) * count + candidate).extract(0usize);
                let labor = load(
                    state,
                    comptime!(Field::LaborMinutes as u32) * count + candidate,
                )
                .extract(0usize);
                let output = load(
                    state,
                    comptime!(Field::OutputTicks as u32) * count + candidate,
                )
                .extract(0usize);
                let inputs = load(
                    state,
                    comptime!(Field::InputTicks as u32) * count + candidate,
                )
                .extract(0usize);
                let first_cost = load(
                    state,
                    comptime!(Field::FirstInputCost as u32) * count + candidate,
                )
                .extract(0usize);
                let second_cost = load(
                    state,
                    comptime!(Field::SecondInputCost as u32) * count + candidate,
                )
                .extract(0usize);
                let packet_cost = plus(
                    times32(wage, labor),
                    times32(inputs, first_cost + second_cost),
                )
                .value;
                let profitable = !equal(sales, zero()) && lt(packet_cost, times32(price, output));
                let ax =
                    actors[ActorLayout::X * (actors.len() / ActorLayout::WORDS) + actor as usize];
                let ay =
                    actors[ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + actor as usize];
                let bx = actors
                    [ActorLayout::X * (actors.len() / ActorLayout::WORDS) + candidate as usize];
                let by = actors
                    [ActorLayout::Y * (actors.len() / ActorLayout::WORDS) + candidate as usize];
                let nearby =
                    max(ax, bx) - min(ax, bx) + max(ay, by) - min(ay, by) <= config.search_radius;
                if action == 0
                    && candidate_kind > 0
                    && status < comptime!(EnterpriseState::Liquidating as u32)
                    && profitable
                    && nearby
                    && !equal(demand, zero())
                    && lt(funds, operating)
                {
                    action = 0x07000000u32 + candidate + 1;
                }
            }
        }
        output[actor as usize] = action;
    }
    action
}

pub(super) fn propose(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    config: &Configuration,
    spatial: &CubeTensor,
    records: RecordBuffers<'_>,
    day: u32,
) {
    let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
    let cube_count = device.count(records.requests.meta.num_elements());
    EnterpriseProposals::launch(
        device,
        cube_count,
        CubeDim::new_1d(device.workgroup),
        ProposalInputLaunch {
            state: argument(state),
            actors: argument(actors),
            config: config.launch(),
            spatial: argument(spatial),
            output: argument(records.requests),
            records: RecordCountBuffersLaunch {
                births: argument(records.births),
                scratch: argument(records.scratch),
                scan: super::parallel::ScanMode::new(device),
            },
            day,
        },
    );
}

pub(super) fn enter(
    state: &mut crate::domain::State,
    scenario: &crate::domain::Scenario,
    requests: &[u32],
) -> anyhow::Result<usize> {
    use crate::domain::{
        Actor,
        enterprise::{EnterpriseModel, Ownership, OwnershipShare},
    };
    use anyhow::Context;
    let entries = requests
        .iter()
        .enumerate()
        .filter(|(_, request)| {
            (1..=Good::COUNT as u32).contains(*request) || matches!(**request >> 24, 8 | 9)
        })
        .map(|(owner, &request)| opportunities::EntryPlan::new(state, scenario, owner, request))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let before = state.actors.len();
    let after = before + entries.len();
    let mut values = vec![0; after * scenario.fields()];
    for field in 0..scenario.fields() {
        values[field * after..field * after + before]
            .copy_from_slice(&state.values[field * before..(field + 1) * before]);
    }
    values.extend_from_slice(&state.values[before * scenario.fields()..]);
    state.values = values;
    let mut actors = state.actors.iter().collect::<Vec<_>>();
    for entry in &entries {
        let identity = state.next_identity;
        state.next_identity = identity
            .checked_add(1)
            .context("enterprise identity capacity exhausted")?;
        actors.push(Actor {
            identity,
            position: entry.position,
            kind: entry.good as u32,
            owner: entry.owner as u32,
            capacity: 1,
        });
    }
    state.actors = actors.into_iter().collect();
    for (offset, entry) in entries.iter().enumerate() {
        let firm = before + offset;
        let owner = entry.owner;
        let cash = state.get(Field::Cash, owner);
        let invested = cash / 2;
        state.set(Field::Cash, owner, cash - invested);
        state.set(Field::Cash, firm, invested);
        match &scenario.config().property {
            Some(property) => {
                state.set(
                    Field::PropertyRegion,
                    firm,
                    property.region(entry.position)? as u64 + 1,
                );
                state.set(
                    Field::PropertyUse,
                    firm,
                    crate::domain::property::PropertyUse::for_actor(&state.actors.at(firm))? as u64,
                );
                state.set(Field::RequestedSpace, firm, 1);
                state.set(
                    Field::ProjectStatus,
                    firm,
                    crate::domain::property::ProjectStatus::Requested as u64,
                );
            }
            None => {
                for (material, quantity) in [(Good::Tools, 2), (Good::Wood, 5)] {
                    state.set(
                        Field::inventory(material),
                        owner,
                        state.get(Field::inventory(material), owner) - quantity,
                    );
                    state.values
                        [(crate::domain::USED_FIELDS + material as usize - 1) * after + firm] =
                        quantity;
                }
                state.set(Field::Used, firm, 7);
                state.set(Field::InstalledCapital, firm, 7);
            }
        }
        state.set(
            Field::ConstructionRemaining,
            firm,
            match scenario.config().property {
                Some(_) => u64::from(crate::domain::property::CONSTRUCTION_WORK_PER_SPACE),
                None => 720,
            },
        );
        state.set(
            Field::EnterpriseStatus,
            firm,
            EnterpriseState::Forming as u64,
        );
        state.set(Field::EquipmentCondition, firm, 1000);
        state.set(Field::Wage, firm, scenario.config().wage.into());
        state.set(Field::Price, firm, entry.price);
        state.set(
            Field::Demand,
            firm,
            entry.recipe.technique.output_ticks.into(),
        );
        state.set(Field::InvestmentFirm, owner, firm as u64 + 1);
        state.set(Field::InvestmentGood, owner, 0);
        state.set_recipe(firm, entry.recipe);
        state.set(Field::FirstInputCost, firm, entry.first_cost);
        state.set(Field::SecondInputCost, firm, entry.second_cost);
        state.set(Field::EnterpriseModel, firm, entry.model as u64);
        if entry.model == EnterpriseModel::Distributor {
            state.set(Field::ReplacementCost, firm, entry.first_cost);
            state.set(Field::StockCoverage, firm, 3);
            state.set(Field::RetailMargin, firm, 250);
        }
        state.retain_basic_recipe(firm);
        state.ownership.push(Ownership::new(Ownership {
            firm: state.actors.at(firm).identity,
            shares: vec![OwnershipShare {
                household: state.actors.at(owner).identity,
                units: 1_000_000,
            }],
        })?);
    }
    Ok(entries.len())
}

pub(super) fn transitions(
    state: &mut crate::domain::State,
    scenario: &crate::domain::Scenario,
    requests: &[u32],
) -> anyhow::Result<()> {
    use crate::domain::{Actor, Position, demography::LivingState};
    use anyhow::{Context, anyhow};
    let splits = requests
        .iter()
        .enumerate()
        .filter(|(_, action)| **action >> 24 == 1)
        .map(|(household, _)| household)
        .collect::<Vec<_>>();
    let before = state.actors.len();
    let after = before + splits.len();
    if after > before {
        let mut values = vec![0; after * scenario.fields()];
        for field in 0..scenario.fields() {
            values[field * after..field * after + before]
                .copy_from_slice(&state.values[field * before..(field + 1) * before]);
        }
        values.extend_from_slice(&state.values[before * scenario.fields()..]);
        state.values = values;
        let mut actors = state.actors.iter().collect::<Vec<_>>();
        for (offset, &household) in splits.iter().enumerate() {
            let identity = state.next_identity;
            state.next_identity = identity
                .checked_add(1)
                .context("household identity capacity exhausted")?;
            actors.push(Actor {
                identity,
                position: actors[household].position,
                kind: 0,
                owner: (before + offset) as u32,
                capacity: 1,
            });
        }
        state.actors = actors.into_iter().collect();
        for (offset, &parent) in splits.iter().enumerate() {
            let household = before + offset;
            let person_identity = state.get(Field::SplitPerson, parent);
            let person = state
                .people
                .iter_mut()
                .find(|person| person.identity == person_identity)
                .context("unknown household formation person")?;
            match person.state {
                LivingState::Living => (),
                _ => Err(anyhow!("deceased person cannot form a household"))?,
            }
            person.household = state.actors.at(household).identity;
            for field in std::iter::once(Field::Cash)
                .chain(Good::ALL.map(Field::inventory))
                .chain([Field::LandPlots, Field::CropWork])
            {
                let assets = state.get(field, parent);
                let share = assets / state.get(Field::LivingMembers, parent).max(2);
                state.set(field, parent, assets - share);
                state.set(field, household, share);
            }
            state.set(
                Field::ParentHousehold,
                household,
                state.actors.at(parent).identity,
            );
            state.set(Field::LivingMembers, household, 1);
            state.set(
                Field::LivingMembers,
                parent,
                state.get(Field::LivingMembers, parent) - 1,
            );
            state.set(Field::Price, household, state.get(Field::Price, parent));
            state.set(Field::EquipmentCondition, household, 1000);
            state.set(Field::SplitPerson, parent, 0);
            if let Some(property) = &scenario.config().property {
                state.set(
                    Field::PropertyRegion,
                    household,
                    property.region(state.actors.at(household).position)? as u64 + 1,
                );
                state.set(
                    Field::PropertyUse,
                    household,
                    crate::domain::property::PropertyUse::Housing as u64,
                );
            }
        }
    }
    let mut heirs = std::collections::HashMap::new();
    for actor in 0..state.actors.len() {
        let parent = state.get(Field::ParentHousehold, actor);
        if state.actors.at(actor).kind == 0
            && parent > 0
            && state.get(Field::LivingMembers, actor) > 0
        {
            heirs.entry(parent).or_insert(actor);
        }
    }
    for (actor, &action) in requests.iter().enumerate() {
        match action >> 24 {
            2 => {
                let destination = (action & 0x00ffffff)
                    .checked_sub(1)
                    .context("missing migration destination")?
                    as usize;
                let position = state
                    .actors
                    .get(destination)
                    .context("unknown migration destination")?
                    .position;
                let cash = state
                    .get(Field::Cash, actor)
                    .checked_sub(100)
                    .context("unfunded migration")?;
                state.set(Field::Cash, actor, cash);
                state.set(
                    Field::Cash,
                    destination,
                    crate::arithmetic::add(state.get(Field::Cash, destination), 100)?,
                );
                let previous = state.actors.at(actor).position;
                state.set(Field::PreviousX, actor, previous.x.into());
                state.set(Field::PreviousY, actor, previous.y.into());
                state.set(Field::MoveX, actor, position.x.into());
                state.set(Field::MoveY, actor, position.y.into());
                let duration = state
                    .actors
                    .at(actor)
                    .position
                    .distance(position)
                    .div_ceil(scenario.config().shipment_speed)
                    .max(1);
                state.set(Field::MoveDue, actor, state.day + u64::from(duration));
            }
            3 => {
                let mut household = state.actors.at(actor);
                household.position = Position {
                    x: state.get(Field::MoveX, actor) as u32,
                    y: state.get(Field::MoveY, actor) as u32,
                };
                state.actors.set(actor, household);
                state.set(Field::MoveDue, actor, 0);
                state.set(Field::MoveTarget, actor, 0);
                if scenario.config().property.is_some() {
                    state.set(Field::LandPlots, actor, 0);
                    state.set(Field::CropWork, actor, 0);
                }
            }
            6 => {
                let mut firm = state.actors.at(actor);
                firm.capacity = firm
                    .capacity
                    .checked_add(state.get(Field::PendingCapacity, actor) as u32)
                    .context("firm capacity exhausted")?;
                state.actors.set(actor, firm);
                state.set(Field::PendingCapacity, actor, 0);
            }
            7 => invest(state, actor, (action & 0x00ffffff) as usize - 1)?,
            4 | 5 => {
                let receiver = match action >> 24 {
                    4 => Some(state.actors.at(actor).owner as usize),
                    _ => heirs.get(&state.actors.at(actor).identity).copied(),
                };
                if action >> 24 == 5 {
                    state.set(
                        Field::EstateSettled,
                        actor,
                        match receiver {
                            Some(_) => 3,
                            None => 2,
                        },
                    );
                }
                let identity = state.actors.at(actor).identity;
                let claims = match &scenario.config().finance {
                    None => true,
                    Some(finance) => {
                        let layout = finance.layout(state.actors.len() * scenario.fields())?;
                        finance
                            .loans
                            .iter()
                            .enumerate()
                            .filter(|(_, loan)| loan.borrower == identity)
                            .all(|(index, _)| {
                                state.values
                                    [layout.loan(crate::domain::finance::LoanField::Status, index)]
                                    == crate::domain::finance::LoanStatus::Repaid as u64
                            })
                    }
                };
                if let (true, Some(receiver)) = (claims, receiver) {
                    match action >> 24 {
                        4 => liquidate(state, scenario, actor)?,
                        _ => transfer_estate(state, scenario, actor, receiver)?,
                    }
                    match action >> 24 {
                        4 => state.set(
                            Field::EnterpriseStatus,
                            actor,
                            EnterpriseState::Closed as u64,
                        ),
                        _ => {
                            state.set(Field::EstateSettled, actor, 1);
                            state.set(Field::InvestmentFirm, actor, 0);
                            let previous = state.actors.at(actor).identity;
                            let successor = state.actors.at(receiver).identity;
                            for ownership in &mut state.ownership {
                                let mut combined = std::collections::BTreeMap::<u64, u32>::new();
                                for share in &ownership.shares {
                                    let household = match share.household == previous {
                                        true => successor,
                                        false => share.household,
                                    };
                                    *combined.entry(household).or_default() += share.units;
                                }
                                ownership.shares = combined
                                    .into_iter()
                                    .map(|(household, units)| {
                                        crate::domain::enterprise::OwnershipShare {
                                            household,
                                            units,
                                        }
                                    })
                                    .collect();
                            }
                            let owned = state
                                .actors
                                .iter()
                                .enumerate()
                                .filter(|(_, firm)| firm.kind > 0 && firm.owner == actor as u32)
                                .map(|(index, _)| index)
                                .collect::<Vec<_>>();
                            for index in owned {
                                let mut firm = state.actors.at(index);
                                firm.owner = receiver as u32;
                                state.actors.set(index, firm);
                            }
                        }
                    }
                }
            }
            _ => (),
        }
    }
    Ok(())
}

fn transfer_estate(
    state: &mut crate::domain::State,
    scenario: &crate::domain::Scenario,
    from: usize,
    to: usize,
) -> anyhow::Result<()> {
    let shared_region = match &scenario.config().property {
        None => true,
        Some(property) => {
            property.region(state.actors.at(from).position)?
                == property.region(state.actors.at(to).position)?
        }
    };
    let fields = std::iter::once(Field::Cash as usize)
        .chain(Good::ALL.map(|good| Field::inventory(good) as usize))
        .chain([
            Field::Transit as usize,
            Field::LandPlots as usize,
            Field::CropWork as usize,
            Field::InstalledCapital as usize,
        ])
        .chain(crate::domain::TRANSIT_FIELDS..crate::domain::TRANSIT_FIELDS + Good::COUNT)
        .chain(crate::domain::BASE_FIELDS..scenario.fields())
        .filter(|&field| {
            shared_region || ![Field::LandPlots as usize, Field::CropWork as usize].contains(&field)
        });
    let count = state.actors.len();
    for field in fields {
        let amount = state.values[field * count + from];
        state.values[field * count + to] =
            crate::arithmetic::add(state.values[field * count + to], amount)?;
        state.values[field * count + from] = 0;
    }
    if !shared_region {
        state.set(Field::LandPlots, from, 0);
        state.set(Field::CropWork, from, 0);
    }
    Ok(())
}

fn liquidate(
    state: &mut crate::domain::State,
    scenario: &crate::domain::Scenario,
    firm: usize,
) -> anyhow::Result<()> {
    use anyhow::Context;
    let ownership = state
        .ownership
        .iter()
        .find(|ownership| ownership.firm == state.actors.at(firm).identity)
        .context("liquidation requires ownership")?
        .clone();
    let mut remaining = 1_000_000u64;
    for share in ownership.shares {
        let receiver = state
            .actors
            .iter()
            .position(|actor| actor.identity == share.household)
            .context("liquidation shareholder is missing")?;
        let count = state.actors.len();
        let fields = std::iter::once(Field::Cash as usize)
            .chain(Good::ALL.map(|good| Field::inventory(good) as usize))
            .chain([
                Field::LandPlots as usize,
                Field::CropWork as usize,
                Field::InstalledCapital as usize,
            ])
            .chain(crate::domain::BASE_FIELDS..scenario.fields());
        for field in fields {
            let balance = state.values[field * count + firm];
            let amount =
                (u128::from(balance) * u128::from(share.units) / u128::from(remaining)) as u64;
            state.values[field * count + firm] = balance - amount;
            state.values[field * count + receiver] =
                crate::arithmetic::add(state.values[field * count + receiver], amount)?;
        }
        for actor in [firm, receiver] {
            let mut total = 0;
            for good in Good::ALL {
                let mut transit = 0;
                for slot in 0..scenario.ring() {
                    let coordinate = state
                        .shipment_coordinate(good, slot, scenario.ring(), actor)
                        .0 as usize;
                    transit = crate::arithmetic::add(transit, state.values[coordinate])?;
                }
                state.values[(crate::domain::TRANSIT_FIELDS + good as usize - 1) * count + actor] =
                    transit;
                total = crate::arithmetic::add(total, transit)?;
            }
            state.set(Field::Transit, actor, total);
        }
        remaining -= u64::from(share.units);
    }
    Ok(())
}

fn invest(state: &mut crate::domain::State, investor: usize, firm: usize) -> anyhow::Result<()> {
    use crate::domain::enterprise::{Ownership, OwnershipShare};
    use anyhow::Context;
    let paid = 5000u64;
    let household_cash = state
        .get(Field::Cash, investor)
        .checked_sub(paid)
        .context("unfunded household investment")?;
    let funds = crate::arithmetic::add(state.get(Field::Cash, firm), paid)?;
    let book = u128::from(funds)
        + u128::from(state.get(Field::InstalledCapital, firm)) * 100
        + u128::from(state.values[state.actors.at(firm).kind as usize * state.actors.len() + firm])
            * u128::from(state.get(Field::Price, firm));
    let units = (u128::from(paid) * 1_000_000 / book.max(1)).clamp(1, 250_000) as u32;
    let owner = state.actors.at(investor).identity;
    let index = state
        .ownership
        .iter()
        .position(|ownership| ownership.firm == state.actors.at(firm).identity)
        .context("investment requires issued shares")?;
    let mut combined = std::collections::BTreeMap::<u64, u32>::new();
    for share in &state.ownership[index].shares {
        let retained = (u64::from(share.units) * u64::from(1_000_000 - units) / 1_000_000) as u32;
        if retained > 0 {
            combined.insert(share.household, retained);
        }
    }
    *combined.entry(owner).or_default() += units;
    let allocated = combined.values().sum::<u32>();
    let founder = state
        .actors
        .at(state.actors.at(firm).owner as usize)
        .identity;
    *combined.entry(founder).or_default() += 1_000_000 - allocated;
    if combined.get(&founder).copied().unwrap_or(0) == 0 {
        combined.insert(founder, 1);
        *combined
            .get_mut(&owner)
            .context("investment shares are missing")? -= 1;
    }
    state.ownership[index] = Ownership::new(Ownership {
        firm: state.actors.at(firm).identity,
        shares: combined
            .into_iter()
            .filter(|(_, units)| *units > 0)
            .map(|(household, units)| OwnershipShare { household, units })
            .collect(),
    })?;
    state.set(Field::Cash, investor, household_cash);
    state.set(Field::Cash, firm, funds);
    state.set(Field::EquityInvested, investor, paid);
    Ok(())
}

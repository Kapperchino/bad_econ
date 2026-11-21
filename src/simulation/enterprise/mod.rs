pub(super) mod acquisitions;
pub(super) mod management;
mod opportunities;
pub(super) mod ownership;
pub(in crate::simulation) use opportunities::EntryQuoteMode;

use super::{
    Device,
    layout::ActorLayout,
    pass::GpuPass,
    registry::records::{
        CountRecords, ProposedRecord, RecordBuffers, RecordCountBuffers, RecordCountBuffersLaunch,
    },
    system::{AsBufferArg, System, gpu_passes},
    words::*,
};
use crate::domain::{
    Field, Good,
    enterprise::{EnterpriseState, EntrepreneurialInterest, EstateStatus},
};
use crate::simulation::configuration::{Configuration, EconomyConfig, Feature};
use burn_wgpu::CubeTensor;
use cubecl::{CubeDim, prelude::*};

const FORM_HOUSEHOLD: u32 = 0x0100_0001;
const SEPARATE_OWNER: u32 = 0x0100_0002;

#[cube]
pub(in crate::simulation) fn founder_shares(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    household: u32,
) -> u32 {
    let pooled = comptime!(config.cohorts == Feature::Enabled)
        && actors[ActorLayout::CAPACITY * (actors.len() / ActorLayout::WORDS) + household as usize]
            > 1;
    let living = load(
        state,
        comptime!(Field::LivingMembers as u32) * config.state_stride + household,
    );
    select(pooled, max(living.extract(0usize), 1), 1)
}

#[derive(CubeType, CubeLaunch)]
struct ProposalInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    spatial: Box<[u32]>,
    output: Box<[u32]>,
    records: RecordCountBuffers,
    day: u32,
    #[cube(comptime)]
    quote_mode: opportunities::EntryQuoteMode,
}

struct EnterpriseProposals;
struct ProposeEnterprises;
struct EntryQuotes;
struct PrepareEntryQuotes;

impl System for EntryQuotes {
    type Input = ProposalInput;
    type Passes = gpu_passes![PrepareEntryQuotes];
}

#[cube]
impl GpuPass<ProposalInput> for PrepareEntryQuotes {
    type Output = ();

    fn execute(mut input: ProposalInput) {
        opportunities::prepare_quotes(
            &mut input.state,
            &input.actors,
            &input.config,
            &input.spatial,
            &mut input.records.scratch,
            input.day,
        );
        crate::simulation::economy::mobility::prepare_quotes(
            &mut input.state,
            &input.actors,
            &input.config,
            &input.spatial,
            &mut input.records.scratch,
            input.day,
        );
    }
}

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
            &input.records.scratch,
            &mut input.output,
            input.day,
            comptime!(input.quote_mode),
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
    quotes: &[u32],
    output: &mut [u32],
    day: u32,
    #[comptime] quote_mode: opportunities::EntryQuoteMode,
) -> u32 {
    let actor = ABSOLUTE_POS as u32;
    let mut action = 0u32;
    if actor < config.actors {
        if comptime!(config.home_mode == Feature::Enabled)
            && actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + actor as usize] == 0
        {
            crate::simulation::economy::enterprises::mobility(
                state, actors, config, spatial, quotes, actor, day, quote_mode,
            );
        }
        let target = load(
            state,
            comptime!(Field::InvestmentGood as u32) * config.state_stride + actor,
        );
        let tools = load(
            state,
            comptime!(Field::Tools as u32) * config.state_stride + actor,
        );
        let wood = load(
            state,
            comptime!(Field::Wood as u32) * config.state_stride + actor,
        );
        let cash = load(
            state,
            comptime!(Field::Cash as u32) * config.state_stride + actor,
        );
        let arrears = load(
            state,
            comptime!(Field::RentArrears as u32) * config.state_stride + actor,
        );
        let household =
            actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + actor as usize] == 0;
        let invested = load(
            state,
            comptime!(Field::InvestmentFirm as u32) * config.state_stride + actor,
        );
        let living = load(
            state,
            comptime!(Field::LivingMembers as u32) * config.state_stride + actor,
        );
        let people = founder_shares(state, actors, config, actor);
        let capital = divide(cash, people);
        let funded = !lt(capital, ticks(10_000, 0))
            && (comptime!(config.property_mode == Feature::Enabled)
                || (!lt(tools, times32(people, 2)) && !lt(wood, times32(people, 5))))
            && equal(arrears, zero());
        let pending_investment =
            comptime!(config.cohorts == Feature::Enabled) && target.extract(0usize) >> 24 == 7;
        let eligible = !pending_investment
            && comptime!(config.home_mode == Feature::Enabled)
            && household
            && (actors[ActorLayout::ENTREPRENEURIAL_INTEREST
                * (actors.len() / ActorLayout::WORDS)
                + actor as usize]
                == comptime!(EntrepreneurialInterest::Interested as u32)
                || (comptime!(config.cohorts == Feature::Enabled) && !equal(target, zero())))
            && !equal(living, zero())
            && equal(invested, zero())
            && equal(arrears, zero())
            && !lt(capital, ticks(10_000, 0));
        let mut proposal = 0;
        let review = !equal(target, zero()) || (day + actor) % 30 == 0;
        if eligible && review {
            proposal = opportunities::discover(
                state, actors, config, spatial, quotes, actor, day, quote_mode,
            );
            let source = select(proposal >> 24 == 8, (proposal & 0x00ffffff) - 1, actor);
            if !crate::simulation::economy::property::authorized(state, config, source, actor) {
                proposal = 0;
            }
            action = select(funded, proposal, 0);
        }
        store(
            state,
            comptime!(Field::InvestmentGood as u32) * config.state_stride + actor,
            ticks(proposal, 0),
        );
        let split = load(
            state,
            comptime!(Field::SplitPerson as u32) * config.state_stride + actor,
        );
        if !equal(split, zero()) && !lt(cash, ticks(5000, 0)) {
            action = FORM_HOUSEHOLD;
        }
        let moving = load(
            state,
            comptime!(Field::MoveDue as u32) * config.state_stride + actor,
        )
        .extract(0usize);
        let destination = load(
            state,
            comptime!(Field::MoveTarget as u32) * config.state_stride + actor,
        )
        .extract(0usize);
        if destination > 0 && moving == 0 {
            let stride = actors.len() / ActorLayout::WORDS;
            let target = destination as usize - 1;
            let x = actors[ActorLayout::X * stride + actor as usize];
            let y = actors[ActorLayout::Y * stride + actor as usize];
            let dx = actors[ActorLayout::X * stride + target];
            let dy = actors[ActorLayout::Y * stride + target];
            let distance = max(x, dx) - min(x, dx) + max(y, dy) - min(y, dy);
            let rural = !equal(
                load(
                    state,
                    comptime!(Field::LandPlots as u32) * config.state_stride + actor,
                ),
                zero(),
            );
            let nearby = rural && distance <= config.commute_speed;
            let migration = select(!lt(cash, ticks(100, 0)), 0x02000000u32 + destination, 0);
            action = select(
                nearby,
                comptime!(crate::simulation::registry::WAGE_DEPENDENCY),
                migration,
            );
        }
        if moving > 0 && moving <= day {
            action = 0x03000000u32;
        }
        let kind = actors[ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + actor as usize];
        let status = load(
            state,
            comptime!(Field::EnterpriseStatus as u32) * config.state_stride + actor,
        )
        .extract(0usize);
        let living = load(
            state,
            comptime!(Field::LivingMembers as u32) * config.state_stride + actor,
        )
        .extract(0usize);
        if kind == 0 && living == 0 && (action <= Good::COUNT as u32 || action >> 24 >= 8) {
            action = 0;
        }
        let estate = load(
            state,
            comptime!(Field::EstateSettled as u32) * config.state_stride + actor,
        )
        .extract(0usize);
        if kind > 0 && status == comptime!(EnterpriseState::Liquidating as u32) {
            action = 0x04000000u32;
        }
        if kind == 0
            && living == 0
            && (estate == comptime!(EstateStatus::Open as u32)
                || (estate == comptime!(EstateStatus::AwaitingSettlement as u32)
                    && (day + actor) % 30 == 0))
        {
            action = 0x05000000u32;
        }
        let pending = load(
            state,
            comptime!(Field::PendingCapacity as u32) * config.state_stride + actor,
        )
        .extract(0usize);
        let construction = load(
            state,
            comptime!(Field::ConstructionRemaining as u32) * config.state_stride + actor,
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
            comptime!(Field::EquityInvested as u32) * config.state_stride + actor,
        );
        if action == 0
            && kind == 0
            && living > 0
            && equal(invested, zero())
            && !lt(capital, ticks(20_000, 0))
            && (day % 30 == 0 || pending_investment)
        {
            let samples = select(pending_investment, 1u32, 16u32);
            for sample in 0..samples {
                let candidate = select(
                    pending_investment,
                    (target.extract(0usize) & 0x00ff_ffff) - 1,
                    actor + day * 17 + sample * 31,
                ) % config.actors;
                let candidate_kind = actors
                    [ActorLayout::KIND * (actors.len() / ActorLayout::WORDS) + candidate as usize];
                let status = load(
                    state,
                    comptime!(Field::EnterpriseStatus as u32) * config.state_stride + candidate,
                )
                .extract(0usize);
                let sales = load(
                    state,
                    comptime!(Field::Sales as u32) * config.state_stride + candidate,
                );
                let demand = load(
                    state,
                    comptime!(Field::Demand as u32) * config.state_stride + candidate,
                );
                let operating = load(
                    state,
                    comptime!(Field::OperatingReserve as u32) * config.state_stride + candidate,
                );
                let funds = load(
                    state,
                    comptime!(Field::Cash as u32) * config.state_stride + candidate,
                );
                let price = load(
                    state,
                    comptime!(Field::Price as u32) * config.state_stride + candidate,
                )
                .extract(0usize);
                let wage = load(
                    state,
                    comptime!(Field::Wage as u32) * config.state_stride + candidate,
                )
                .extract(0usize);
                let labor = load(
                    state,
                    comptime!(Field::LaborMinutes as u32) * config.state_stride + candidate,
                )
                .extract(0usize);
                let output = load(
                    state,
                    comptime!(Field::OutputTicks as u32) * config.state_stride + candidate,
                )
                .extract(0usize);
                let inputs = load(
                    state,
                    comptime!(Field::InputTicks as u32) * config.state_stride + candidate,
                )
                .extract(0usize);
                let first_cost = load(
                    state,
                    comptime!(Field::FirstInputCost as u32) * config.state_stride + candidate,
                )
                .extract(0usize);
                let second_cost = load(
                    state,
                    comptime!(Field::SecondInputCost as u32) * config.state_stride + candidate,
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
        if action == 0
            && day % 30 == 0
            && kind > 0
            && status == comptime!(EnterpriseState::Operating as u32)
        {
            let reserve = load(
                state,
                comptime!(Field::OperatingReserve as u32) * config.state_stride + actor,
            );
            action = opportunities::purchase(
                state,
                actors,
                config,
                spatial,
                actor,
                day,
                positive(
                    cash,
                    maximum(
                        reserve,
                        select(
                            kind == comptime!(Good::Housing as u32),
                            zero(),
                            ticks(config.reserve_low, config.reserve_high),
                        ),
                    ),
                ),
            );
        }
        let ownership_change = (action > 0 && action <= Good::COUNT as u32)
            || (action >> 24 >= 7 && action >> 24 <= 9);
        if people > 1 && ownership_change {
            store(
                state,
                comptime!(Field::InvestmentGood as u32) * config.state_stride + actor,
                ticks(action, 0),
            );
            let person = load(
                state,
                comptime!(Field::CohortDelegate as u32) * config.state_stride + actor,
            );
            store(
                state,
                comptime!(Field::SplitPerson as u32) * config.state_stride + actor,
                person,
            );
            action = select(equal(person, zero()), 0, SEPARATE_OWNER);
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
    propose_quotes(
        device,
        state,
        actors,
        config,
        spatial,
        records,
        day,
        opportunities::EntryQuoteMode::Cached,
    );
}

fn propose_quotes(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    config: &Configuration,
    spatial: &CubeTensor,
    records: RecordBuffers<'_>,
    day: u32,
    quote_mode: opportunities::EntryQuoteMode,
) {
    let cube_count = device.count(records.requests.meta.num_elements());
    let input = || ProposalInputLaunch {
        state: state.buffer_arg(),
        actors: actors.buffer_arg(),
        config: config.launch(),
        spatial: spatial.buffer_arg(),
        output: records.requests.buffer_arg(),
        records: RecordCountBuffersLaunch {
            births: records.births.buffer_arg(),
            scratch: records.scratch.buffer_arg(),
            scan: super::parallel::ScanMode::new(device),
        },
        day,
        quote_mode,
    };
    EntryQuotes::launch(
        device,
        cube_count.clone(),
        CubeDim::new_1d(device.workgroup),
        input(),
    );
    EnterpriseProposals::launch(
        device,
        cube_count,
        CubeDim::new_1d(device.workgroup),
        input(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{Experiment, HouseholdLiquidity, Scenario};
    use crate::simulation::{Simulation, decode, encode};

    #[derive(Clone, Copy, Debug)]
    enum EntryInputs {
        Stocked,
        Missing,
    }

    #[test]
    fn cached_migration_needs_preserve_destinations_and_compacted_records_on_gpu()
    -> anyhow::Result<()> {
        #[derive(Clone, Copy, Debug)]
        enum Supplies {
            Available,
            MissingFood,
            MissingHeat,
        }

        let device = crate::simulation::tests::device(64)?;
        let mut config = Scenario::industrial(257, 42, Experiment::Baseline)?
            .config()
            .clone();
        config.entrepreneurship.interested_per_mille = 0;
        config.commute_speed = 1;
        config.delivery_penalty = 10_000;
        config.household_liquidity = HouseholdLiquidity::Uniform;
        config.household_cash = 1_000_000;
        let scenario = Scenario::new(config)?;
        for day in [1, 20, 30, 120] {
            for supplies in [
                Supplies::Available,
                Supplies::MissingFood,
                Supplies::MissingHeat,
            ] {
                let expected = Simulation::new(scenario.clone(), device.clone())?;
                let actual = Simulation::new(scenario.clone(), device.clone())?;
                let mut initial = scenario.initial()?;
                let firms = initial.actors.firm_indices().collect::<Vec<_>>();
                let urban = firms
                    .iter()
                    .copied()
                    .filter(|&firm| initial.actors.at(firm).kind == Good::Bread as u32)
                    .collect::<Vec<_>>();
                for owner in 0..scenario.config().households as usize {
                    let position = initial.actors.at(owner).position;
                    let employer = urban
                        .iter()
                        .copied()
                        .max_by_key(|&firm| position.distance(initial.actors.at(firm).position))
                        .expect("urban employer");
                    let rural = owner >= 64;
                    let employed = !(128..192).contains(&owner);
                    initial.set(
                        Field::Employer,
                        owner,
                        if employed { employer as u64 + 1 } else { 0 },
                    );
                    initial.set(Field::Employed, owner, u64::from(employed));
                    initial.set(Field::LivingMembers, owner, 1);
                    initial.set(Field::LandPlots, owner, u64::from(rural));
                    initial.set(Field::IncomeForecast, owner, 4_800_000);
                    initial.set(
                        Field::FoodPrice,
                        owner,
                        if owner >= 192 { 9_000 } else { 100 },
                    );
                    initial.set(Field::NutritionNeed, owner, 1);
                    initial.set(Field::HeatingNeed, owner, 2);
                    initial.set(Field::MoveDue, owner, 0);
                    initial.set(Field::MoveTarget, owner, 0);
                    initial.set(Field::Shortage, owner, 0);
                    initial.set(Field::EquityInvested, owner, 1);
                }
                for firm in firms {
                    initial.set(
                        Field::EnterpriseStatus,
                        firm,
                        EnterpriseState::Operating as u64,
                    );
                    initial.set(Field::Cash, firm, 1_000_000);
                    initial.set(Field::Wage, firm, 10);
                    initial.set(Field::LaborMinutes, firm, 480);
                    initial.set(Field::Slots, firm, 100);
                    initial.set(Field::Price, firm, 1);
                    initial.set(
                        Field::Bread,
                        firm,
                        match supplies {
                            Supplies::MissingFood => 0,
                            _ => 1_000_000,
                        },
                    );
                    for field in [Field::Coal, Field::Wood] {
                        initial.set(
                            field,
                            firm,
                            match supplies {
                                Supplies::MissingHeat => 0,
                                _ => 1_000_000,
                            },
                        );
                    }
                }
                for (simulation, mode) in [
                    (&expected, EntryQuoteMode::Sampled),
                    (&actual, EntryQuoteMode::Cached),
                ] {
                    device.write(
                        &simulation.state,
                        bytemuck::cast_slice(&encode(&crate::simulation::tensor::state_values(
                            &initial.values,
                            initial.actors.len(),
                            simulation.config.state_stride as usize,
                        ))),
                    );
                    let records = RecordBuffers {
                        requests: &simulation.enterprise_requests,
                        births: simulation.demography.birth_buffer(),
                        scratch: &simulation.scratch,
                    };
                    propose_quotes(
                        &device,
                        &simulation.state,
                        &simulation.actors,
                        &simulation.config,
                        &simulation.spatial,
                        records,
                        day,
                        mode,
                    );
                }
                let values = device.read(&actual.state)?;
                assert_eq!(
                    values,
                    device.read(&expected.state)?,
                    "day {day}, {supplies:?}"
                );
                let values = decode(&values);
                let start = Field::MoveTarget as usize * actual.config.state_stride as usize;
                let targets = &values[start..start + scenario.config().households as usize];
                assert!(targets[128..].iter().all(|&target| target == 0));
                let farm_moves = (64..128)
                    .filter(|&owner| targets[owner] > 0)
                    .collect::<Vec<_>>();
                assert!(
                    farm_moves
                        .iter()
                        .all(|&owner| (day as usize + owner).is_multiple_of(30))
                );
                match supplies {
                    Supplies::Available => {
                        assert!(targets[..64].iter().any(|&target| target > 0));
                        assert!(!farm_moves.is_empty(), "day {day}");
                    }
                    _ => assert!(targets.iter().all(|&target| target == 0)),
                }
                let pending = |simulation: &Simulation| {
                    crate::simulation::registry::records::PendingRecords::read(
                        &device,
                        RecordBuffers {
                            requests: &simulation.enterprise_requests,
                            births: simulation.demography.birth_buffer(),
                            scratch: &simulation.scratch,
                        },
                        &simulation.searches.requests,
                    )
                };
                assert_eq!(pending(&actual)?.records, pending(&expected)?.records);
            }
        }
        Ok(())
    }

    #[test]
    fn cached_entry_quotes_preserve_sampled_proposals_and_household_state_on_gpu()
    -> anyhow::Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let mut config = Scenario::industrial(3000, 42, Experiment::Baseline)?
            .config()
            .clone();
        config.entrepreneurship.interested_per_mille = 1000;
        config.household_liquidity = HouseholdLiquidity::Uniform;
        config.household_cash = 1_000_000;
        let scenario = Scenario::new(config)?;
        for (day, inputs) in [1, 20, 30, 120].into_iter().flat_map(|day| {
            [EntryInputs::Stocked, EntryInputs::Missing]
                .into_iter()
                .map(move |inputs| (day, inputs))
        }) {
            let expected = Simulation::new(scenario.clone(), device.clone())?;
            let actual = Simulation::new(scenario.clone(), device.clone())?;
            let mut initial = scenario.initial()?;
            for owner in 0..scenario.config().households as usize {
                initial.set(
                    Field::LivingMembers,
                    owner,
                    u64::from(scenario.config().demography.initial_household_members),
                );
                initial.set(Field::InvestmentFirm, owner, 0);
                initial.set(Field::InvestmentGood, owner, 1);
                initial.set(Field::Tools, owner, 2);
                initial.set(Field::Wood, owner, 10);
            }
            for firm in initial.actors.firm_indices().collect::<Vec<_>>() {
                let good = Good::from_index(initial.actors.at(firm).kind)?;
                initial.set(
                    Field::EnterpriseStatus,
                    firm,
                    match good {
                        Good::Bread | Good::Flour => EnterpriseState::Operating as u64,
                        _ => EnterpriseState::Dormant as u64,
                    },
                );
                match good {
                    Good::Bread => {
                        initial.set(
                            Field::Price,
                            firm,
                            u64::from(scenario.config().maximum_price),
                        );
                        initial.set(Field::Bread, firm, 0);
                    }
                    Good::Flour => {
                        initial.set(Field::Price, firm, 1);
                        initial.set(
                            Field::Flour,
                            firm,
                            match inputs {
                                EntryInputs::Stocked => 1_000_000,
                                EntryInputs::Missing => 0,
                            },
                        );
                    }
                    _ => (),
                }
            }
            for (simulation, quote_mode) in [
                (&expected, opportunities::EntryQuoteMode::Sampled),
                (&actual, opportunities::EntryQuoteMode::Cached),
            ] {
                device.write(
                    &simulation.state,
                    bytemuck::cast_slice(&encode(&initial.values)),
                );
                propose_quotes(
                    &device,
                    &simulation.state,
                    &simulation.actors,
                    &simulation.config,
                    &simulation.spatial,
                    RecordBuffers {
                        requests: &simulation.enterprise_requests,
                        births: simulation.demography.birth_buffer(),
                        scratch: &simulation.scratch,
                    },
                    day,
                    quote_mode,
                );
            }
            let state = device.read(&actual.state)?;
            assert_eq!(
                state,
                device.read(&expected.state)?,
                "day {day}, {inputs:?}"
            );
            assert_eq!(
                device.read(&actual.enterprise_requests)?,
                device.read(&expected.enterprise_requests)?,
                "day {day}, {inputs:?}"
            );
            let values = decode(&state);
            let stride = actual.config.state_stride as usize;
            let producers = (0..scenario.config().households as usize)
                .filter(|&owner| values[Field::InvestmentGood as usize * stride + owner] >> 24 == 8)
                .collect::<Vec<_>>();
            match inputs {
                EntryInputs::Stocked => {
                    assert!(!producers.is_empty(), "day {day}: no producer discoveries");
                    for owner in producers {
                        assert_eq!(values[Field::ReplacementCost as usize * stride + owner], 1);
                        assert_eq!(values[Field::SecondInputCost as usize * stride + owner], 0);
                    }
                }
                EntryInputs::Missing => assert!(
                    producers.is_empty(),
                    "day {day}: unavailable input was accepted"
                ),
            }
        }
        Ok(())
    }
}

fn resize_values(state: &mut crate::domain::State, fields: usize, actors: usize) {
    let before = state.actors.len();
    let mut values = vec![0; actors * fields];
    for field in 0..fields {
        values[field * actors..field * actors + before]
            .copy_from_slice(&state.values[field * before..(field + 1) * before]);
    }
    values.extend_from_slice(&state.values[before * fields..]);
    state.values = values;
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
    resize_values(state, scenario.fields(), after);
    let actors = entries
        .iter()
        .map(|entry| {
            let identity = state.next_identity;
            state.next_identity = identity
                .checked_add(1)
                .context("enterprise identity capacity exhausted")?;
            Ok(Actor {
                identity,
                position: entry.position,
                kind: entry.good as u32,
                owner: entry.owner as u32,
                capacity: 1,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    state.actors.extend(actors);
    for (offset, entry) in entries.iter().enumerate() {
        let firm = before + offset;
        let owner = entry.owner;
        let cash = state.get(Field::Cash, owner);
        let invested = cash / 2;
        state.set(Field::Cash, owner, cash - invested);
        state.set(Field::Cash, firm, invested);
        state.set(Field::EnterpriseModel, firm, entry.model as u64);
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
                    crate::domain::property::PropertyUse::for_site(state, firm)? as u64,
                );
                state.set(Field::RequestedPlots, firm, 1);
                state.set(Field::WorkersPerPlot, firm, 1);
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
                Some(_) => u64::from(crate::domain::property::CONSTRUCTION_WORK_PER_PLOT),
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

pub(in crate::simulation) struct HouseholdFormation {
    pub parent: usize,
    pub household: usize,
    pub person: u64,
}

pub(in crate::simulation) fn form_households(
    state: &mut crate::domain::State,
    scenario: &crate::domain::Scenario,
    requests: &[u32],
) -> anyhow::Result<Vec<HouseholdFormation>> {
    use crate::domain::Actor;
    use anyhow::Context;
    let splits = requests
        .iter()
        .enumerate()
        .filter(|(_, action)| **action >> 24 == 1)
        .map(|(household, _)| household)
        .collect::<Vec<_>>();
    let mut formations = Vec::with_capacity(splits.len());
    let before = state.actors.len();
    let after = before + splits.len();
    if after > before {
        resize_values(state, scenario.fields(), after);
        let actors = splits
            .iter()
            .enumerate()
            .map(|(offset, &household)| {
                let identity = state.next_identity;
                state.next_identity = identity
                    .checked_add(1)
                    .context("household identity capacity exhausted")?;
                Ok(Actor {
                    identity,
                    position: state.actors.at(household).position,
                    kind: 0,
                    owner: (before + offset) as u32,
                    capacity: 1,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        state.actors.extend(actors);
        for (offset, &parent) in splits.iter().enumerate() {
            let household = before + offset;
            let person_identity = state.get(Field::SplitPerson, parent);
            formations.push(HouseholdFormation {
                parent,
                household,
                person: person_identity,
            });
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
            if let Some(band) = [Field::YoungAdults, Field::OlderAdults]
                .into_iter()
                .find(|&field| state.get(field, parent) > 0)
            {
                let women = match band {
                    Field::YoungAdults => Field::FemaleYoungAdults,
                    _ => Field::FemaleOlderAdults,
                };
                let female = u64::from(state.get(women, parent) * 2 >= state.get(band, parent));
                let retained = state.get(women, parent) - female;
                let mother = u64::from(
                    band == Field::YoungAdults
                        && state.get(Field::RecoveringMothers, parent) > retained,
                );
                for (field, amount) in [
                    (band, 1),
                    (women, female),
                    (Field::RecoveringMothers, mother),
                ] {
                    state.set(field, parent, state.get(field, parent) - amount);
                    state.set(field, household, amount);
                }
                state.set(Field::EligibleWorkers, household, 1);
                for field in [
                    Field::CohortFamineExposure,
                    Field::CohortColdExposure,
                    Field::Fatigue,
                ] {
                    state.set(field, household, state.get(field, parent));
                }
            }
            if requests[parent] == SEPARATE_OWNER {
                state.set(
                    Field::InvestmentGood,
                    household,
                    state.get(Field::InvestmentGood, parent),
                );
                state.set(Field::InvestmentGood, parent, 0);
            }
            let remaining = state.get(Field::LivingMembers, parent) - 1;
            state.set(Field::LivingMembers, parent, remaining);
            state.set(
                Field::HousingAvailable,
                parent,
                state.get(Field::HousingAvailable, parent).min(remaining),
            );
            state.set(Field::Price, household, state.get(Field::Price, parent));
            state.set(
                Field::FoodPrice,
                household,
                state.get(Field::FoodPrice, parent),
            );
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
    Ok(formations)
}

pub(super) fn transitions(
    state: &mut crate::domain::State,
    scenario: &crate::domain::Scenario,
    requests: &[u32],
) -> anyhow::Result<()> {
    transitions_in_rows(state, scenario, requests, |row| row)
}

pub(super) fn transitions_in_rows(
    state: &mut crate::domain::State,
    scenario: &crate::domain::Scenario,
    requests: &[u32],
    global_row: impl Fn(usize) -> usize,
) -> anyhow::Result<()> {
    use crate::domain::{Position, demography::LivingState};
    use anyhow::{Context, anyhow};
    let formations = form_households(state, scenario, requests)?;
    if !state.people.is_empty() {
        for formation in formations {
            let person = state
                .people
                .iter_mut()
                .find(|person| person.identity == formation.person)
                .context("unknown household formation person")?;
            match person.state {
                LivingState::Living => (),
                _ => Err(anyhow!("deceased person cannot form a household"))?,
            }
            person.household = state.actors.at(formation.household).identity;
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
                if scenario.config().property.is_some()
                    || scenario.config().economy == crate::domain::Economy::Industrial
                {
                    state.set(Field::LandPlots, actor, 0);
                    state.set(Field::CropWork, actor, 0);
                }
            }
            10 => {
                state.set(Field::MoveTarget, actor, 0);
                state.set(Field::LandPlots, actor, 0);
                state.set(Field::CropWork, actor, 0);
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
                            Some(_) => EstateStatus::AwaitingSettlement as u64,
                            None => EstateStatus::Unclaimed as u64,
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
                            state.set(
                                Field::EstateSettled,
                                actor,
                                EstateStatus::Transferred as u64,
                            );
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
                            for index in 0..state.actors.len() {
                                let mut firm = state.actors.at(index);
                                match (firm.kind, firm.owner) {
                                    (1.., owner) if owner == actor as u32 => {
                                        firm.owner = receiver as u32;
                                        state.actors.set(index, firm);
                                    }
                                    _ => (),
                                }
                            }
                        }
                    }
                }
            }
            _ => (),
        }
    }
    acquisitions::settle(state, scenario, requests, global_row)
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
            Field::RoadFreight as usize,
            Field::CartagePayable as usize,
            Field::LandPlots as usize,
            Field::CropWork as usize,
            Field::InstalledCapital as usize,
        ])
        .chain(crate::domain::TRANSIT_FIELDS..crate::domain::TRANSIT_FIELDS + Good::COUNT)
        .chain(
            crate::domain::BASE_FIELDS
                ..crate::domain::BASE_FIELDS + 2 * scenario.ring() * Good::COUNT,
        )
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
            .chain(
                crate::domain::BASE_FIELDS
                    ..crate::domain::BASE_FIELDS + 2 * scenario.ring() * Good::COUNT,
            );
        for field in fields {
            let balance = state.values[field * count + firm];
            let amount =
                (u128::from(balance) * u128::from(share.units) / u128::from(remaining)) as u64;
            state.values[field * count + firm] = balance - amount;
            state.values[field * count + receiver] =
                crate::arithmetic::add(state.values[field * count + receiver], amount)?;
        }
        for actor in [firm, receiver] {
            let mut awaiting = 0;
            let mut payable = 0;
            let mut total = 0;
            for good in Good::ALL {
                let mut transit = 0;
                for slot in 0..scenario.ring() {
                    let coordinate = state
                        .shipment_coordinate(good, slot, scenario.ring(), actor)
                        .0 as usize;
                    let waiting = state.values[coordinate + Good::COUNT * scenario.ring() * count];
                    awaiting = crate::arithmetic::add(awaiting, waiting)?;
                    let rate = slot as u64
                        * u64::from(crate::domain::railway::CART_BILLABLE_MINUTES_PER_UNIT_DAY)
                        * u64::from(scenario.config().wage);
                    payable =
                        crate::arithmetic::add(payable, crate::arithmetic::mul(waiting, rate)?)?;
                    transit = crate::arithmetic::add(
                        transit,
                        crate::arithmetic::add(state.values[coordinate], waiting)?,
                    )?;
                }
                state.values[(crate::domain::TRANSIT_FIELDS + good as usize - 1) * count + actor] =
                    transit;
                total = crate::arithmetic::add(total, transit)?;
            }
            state.set(Field::Transit, actor, total);
            state.set(Field::RoadFreight, actor, awaiting);
            state.set(Field::CartagePayable, actor, payable);
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
    let sites = state.company_sites(state.actors.at(firm).identity);
    let book = u128::from(paid)
        + state
            .actors
            .iter()
            .enumerate()
            .filter(|(_, actor)| sites.contains(&actor.identity))
            .map(|(site, actor)| {
                u128::from(state.get(Field::Cash, site))
                    + u128::from(state.get(Field::InstalledCapital, site)) * 100
                    + u128::from(state.values[actor.kind as usize * state.actors.len() + site])
                        * u128::from(state.get(Field::Price, site))
            })
            .sum::<u128>();
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
    let ownership = Ownership::new(Ownership {
        firm: state.actors.at(firm).identity,
        shares: combined
            .into_iter()
            .filter(|(_, units)| *units > 0)
            .map(|(household, units)| OwnershipShare { household, units })
            .collect(),
    })?;
    state.set_company_ownership(ownership);
    state.set(Field::Cash, investor, household_cash);
    state.set(Field::Cash, firm, funds);
    state.set(Field::EquityInvested, investor, paid);
    Ok(())
}

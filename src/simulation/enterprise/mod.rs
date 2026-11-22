use crate::simulation::DeviceEconomy;
use crate::simulation::state::*;
pub(super) mod acquisitions;
mod formation;
pub(super) mod management;
mod opportunities;
pub(super) mod ownership;
mod settlement;
#[cfg(test)]
mod tests;
mod transitions;

pub(super) use formation::{HouseholdFormation, enter, form_households};
pub(in crate::simulation) use opportunities::EntryQuoteMode;
pub(super) use transitions::{transitions, transitions_in_rows};

use super::{
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
use crate::simulation::configuration::{EconomyConfig, Feature};
use burn_wgpu::CubeTensor;
use cubecl::{CubeDim, prelude::*};

const FORM_HOUSEHOLD: u32 = 0x0100_0001;
const SEPARATE_OWNER: u32 = 0x0100_0002;

#[cube]
pub(in crate::simulation) fn founder_shares(economy: &EconomyView<'_>, household: u32) -> u32 {
    let config = economy.ledger.config;
    let pooled =
        comptime!(config.cohorts == Feature::Enabled) && economy.actors.capacity(household) > 1;
    let living = economy.ledger.value(Field::LivingMembers, household);
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
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
        opportunities::prepare_quotes(
            &economy,
            &input.spatial,
            &mut input.records.scratch,
            input.day,
        );
        crate::simulation::economy::mobility::prepare_quotes(
            &mut economy,
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
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
        let request = requests(
            &mut economy,
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
    economy: &mut EconomyView<'_>,
    spatial: &[u32],
    quotes: &[u32],
    output: &mut [u32],
    day: u32,
    #[comptime] quote_mode: opportunities::EntryQuoteMode,
) -> u32 {
    let config = economy.ledger.config;
    let actor = ABSOLUTE_POS as u32;
    let mut action = 0u32;
    if actor < config.actors {
        if comptime!(config.home_mode == Feature::Enabled) && economy.actors.kind(actor).bits() == 0
        {
            crate::simulation::economy::enterprises::mobility(
                economy, spatial, quotes, actor, day, quote_mode,
            );
        }
        let target = economy.ledger.value(Field::InvestmentGood, actor);
        let tools = economy.ledger.value(Field::Tools, actor);
        let wood = economy.ledger.value(Field::Wood, actor);
        let cash = economy.ledger.value(Field::Cash, actor);
        let arrears = economy.ledger.value(Field::RentArrears, actor);
        let household = economy.actors.kind(actor).bits() == 0;
        let invested = economy.ledger.value(Field::InvestmentFirm, actor);
        let living = economy.ledger.value(Field::LivingMembers, actor);
        let people = founder_shares(&*economy, actor);
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
            && (economy.actors.entrepreneurial_interest(actor)
                == comptime!(EntrepreneurialInterest::Interested as u32)
                || (comptime!(config.cohorts == Feature::Enabled) && !equal(target, zero())))
            && !equal(living, zero())
            && equal(invested, zero())
            && equal(arrears, zero())
            && !lt(capital, ticks(10_000, 0));
        let mut proposal = 0;
        let review = !equal(target, zero()) || (day + actor) % 30 == 0;
        if eligible && review {
            proposal = opportunities::discover(economy, spatial, quotes, actor, day, quote_mode);
            let source = select(proposal >> 24 == 8, (proposal & 0x00ffffff) - 1, actor);
            if !crate::simulation::economy::property::authorized(&economy.ledger, source, actor) {
                proposal = 0;
            }
            action = select(funded, proposal, 0);
        }
        economy
            .ledger
            .set(Field::InvestmentGood, actor, ticks(proposal, 0));
        let split = economy.ledger.value(Field::SplitPerson, actor);
        if !equal(split, zero()) && !lt(cash, ticks(5000, 0)) {
            action = FORM_HOUSEHOLD;
        }
        let moving = economy.ledger.number(Field::MoveDue, actor);
        let destination = economy.ledger.number(Field::MoveTarget, actor);
        if destination > 0 && moving == 0 {
            let distance = economy.actors.distance(actor, destination - 1);
            let rural = !equal(economy.ledger.value(Field::LandPlots, actor), zero());
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
        let kind = economy.actors.kind(actor).bits();
        let status = economy.ledger.number(Field::EnterpriseStatus, actor);
        let living = economy.ledger.number(Field::LivingMembers, actor);
        if kind == 0 && living == 0 && (action <= Good::COUNT as u32 || action >> 24 >= 8) {
            action = 0;
        }
        let estate = economy.ledger.number(Field::EstateSettled, actor);
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
        let pending = economy.ledger.number(Field::PendingCapacity, actor);
        let construction = economy.ledger.number(Field::ConstructionRemaining, actor);
        if kind > 0
            && status == comptime!(EnterpriseState::Operating as u32)
            && pending > 0
            && construction == 0
        {
            action = 0x06000000u32 + pending;
        }
        let invested = economy.ledger.value(Field::EquityInvested, actor);
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
                let candidate_kind = economy.actors.kind(candidate).bits();
                let status = economy.ledger.number(Field::EnterpriseStatus, candidate);
                let sales = economy.ledger.value(Field::Sales, candidate);
                let demand = economy.ledger.value(Field::Demand, candidate);
                let operating = economy.ledger.value(Field::OperatingReserve, candidate);
                let funds = economy.ledger.value(Field::Cash, candidate);
                let price = economy.ledger.number(Field::Price, candidate);
                let wage = economy.ledger.number(Field::Wage, candidate);
                let labor = economy.ledger.number(Field::LaborMinutes, candidate);
                let output = economy.ledger.number(Field::OutputTicks, candidate);
                let inputs = economy.ledger.number(Field::InputTicks, candidate);
                let first_cost = economy.ledger.number(Field::FirstInputCost, candidate);
                let second_cost = economy.ledger.number(Field::SecondInputCost, candidate);
                let packet_cost = plus(
                    times32(wage, labor),
                    times32(inputs, first_cost + second_cost),
                )
                .value;
                let profitable = !equal(sales, zero()) && lt(packet_cost, times32(price, output));
                let nearby = economy.actors.distance(actor, candidate) <= config.search_radius;
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
            let reserve = economy.ledger.value(Field::OperatingReserve, actor);
            action = opportunities::purchase(
                &*economy,
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
            economy
                .ledger
                .set(Field::InvestmentGood, actor, ticks(action, 0));
            let person = economy.ledger.value(Field::CohortDelegate, actor);
            economy.ledger.set(Field::SplitPerson, actor, person);
            action = select(equal(person, zero()), 0, SEPARATE_OWNER);
        }
        output[actor as usize] = action;
    }
    action
}

pub(super) fn propose(
    economy: DeviceEconomy<'_>,
    spatial: &CubeTensor,
    records: RecordBuffers<'_>,
    day: u32,
) {
    propose_quotes(
        economy,
        spatial,
        records,
        day,
        opportunities::EntryQuoteMode::Cached,
    );
}

fn propose_quotes(
    DeviceEconomy {
        device,
        state,
        actors,
        config,
    }: DeviceEconomy<'_>,
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

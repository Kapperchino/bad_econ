pub(super) mod markets;
pub(super) mod scheduler;

use super::{
    flags::{
        ActorKind, COMPLETED_WORD, CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, EventStatus, FaultFlags,
        InputMarket, KernelActorKind, KernelAuthorization, KernelEventStatus, KernelFaults,
        MarketFlag, PROGRESS_WORD,
    },
    households::{KernelFatigue, KernelObservation, ObservationKind, next_fatigue, observed_blend},
    layout::*,
    markets::{LATCH_WORD, RANGE_SLOTS_WORD},
    negotiation::{NegotiationRequest, NegotiationStatus, legal_grid_rate, negotiate},
    parallel::{
        self, Reduction,
        workgroup::{self, TickTree, WorkgroupTree},
    },
    parameters::{Discovery, EconomyParameters},
    pass::GpuPass,
    payroll::PayrollAccrual,
    policy::*,
    scheduler::apply,
    state::*,
    system::{AsBufferArg, System, gpu_passes},
    words::*,
};
use crate::simulation::configuration::{EconomyConfig, Feature};
use crate::{
    domain::{
        BASE_FIELDS, CONSUMPTION_FIELDS, Field, Good, PRODUCTION_FIELDS, Phase, TRANSIT_FIELDS,
        USED_FIELDS,
    },
    ledger::{AuthorizationFlags, EMPTY, LEG_WORDS, LegLayout, MAX_LEGS},
};
use cubecl::prelude::*;
pub(super) mod audit;
pub(super) mod batch;
pub(super) mod cartage;
mod clearing;
mod consumption;
mod employment;
pub(super) mod enterprises;
mod events;
pub(super) mod farming;
mod feedback;
pub(in crate::simulation) mod food;
mod groups;
mod household_production;
pub(super) mod indexing;
pub(in crate::simulation) mod landlords;
pub(super) mod mobility;
mod observation;
pub(super) mod opening;
mod phases;
mod pricing;
pub(in crate::simulation) mod property;
mod proposals;
pub(in crate::simulation) mod railway;
mod rent;
pub(super) mod search;
pub(super) mod sorting;
mod systems;
use super::scheduler::Posting;
use clearing::{PeerBalances, PeerOffer, actor_settlement};
use events::{EventStorage, construct};
use proposals::{InputDemand, trial_packet};
pub(super) use systems::{EconomyInputLaunch, EconomySystem};

#[cube]
fn add_checked(control: &[Atomic<u32>], a: Ticks, b: Ticks) -> Ticks {
    let result = plus(a, b);
    result.faults.record(control);
    result.value
}

#[cube]
fn multiply(control: &[Atomic<u32>], a: Ticks, b: u32) -> Ticks {
    let result = times(a, b);
    result.faults.record(control);
    result.value
}

#[cube]
fn scaled(control: &[Atomic<u32>], value: Ticks, numerator: u32, denominator: u32) -> Ticks {
    let result = ratio(value, numerator, denominator);
    result.faults.record(control);
    result.value
}

#[enumflags2::bitflags]
#[repr(u64)]
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum EconomyStage {
    ClearFeedback,
    RecordFeedback,
    ApplyFeedback,
    IndexCount,
    IndexOffsets,
    IndexFill,
    IndexPrices,
    Local,
    FirmRent,
    HouseholdRent,
    Propose,
    MarketActivity,
    Construct,
    Summarize,
    Inspect,
    GroupInitialize,
    GroupSort,
    GroupMerge,
    GroupExecute,
    GroupRemainder,
    GroupPrepare,
    GroupFinalize,
    FinalizeFixedWages,
    SettleInputTile,
    MarketStart,
    InputMarketStart,
    InputMarketPrepare,
    InputMarketRepeat,
    MarketBoundary,
    GroupRanges,
    SupplierRanges,
    ProposeCompact,
    ProposeRetainedCompact,
    GroupCompactFill,
    GroupRadixLocal,
    GroupRadixScatter,
    InputQuotes,
}

#[cube]
fn run_stage(
    economy: &mut EconomyView<'_>,
    spatial: &mut [u32],
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    #[comptime] stage: EconomyStage,
) {
    let config = economy.ledger.config;
    let id = ABSOLUTE_POS as u32;
    let guarded = comptime!(
        parameters.discovery == Discovery::Indexed
            && matches!(
                stage,
                EconomyStage::IndexCount
                    | EconomyStage::IndexOffsets
                    | EconomyStage::IndexFill
                    | EconomyStage::IndexPrices
                    | EconomyStage::Propose
                    | EconomyStage::GroupInitialize
                    | EconomyStage::GroupPrepare
                    | EconomyStage::GroupSort
                    | EconomyStage::GroupMerge
                    | EconomyStage::SupplierRanges
                    | EconomyStage::GroupRanges
                    | EconomyStage::ProposeCompact
                    | EconomyStage::ProposeRetainedCompact
                    | EconomyStage::GroupCompactFill
                    | EconomyStage::GroupRadixLocal
                    | EconomyStage::GroupRadixScatter
                    | EconomyStage::GroupExecute
                    | EconomyStage::GroupRemainder
                    | EconomyStage::GroupFinalize
                    | EconomyStage::FinalizeFixedWages
            )
    );
    if !guarded || control[LATCH_WORD].load() == 0 {
        match stage {
            EconomyStage::InputQuotes => {
                pricing::refresh(economy, spatial, control, parameters, id)
            }
            EconomyStage::MarketStart
            | EconomyStage::InputMarketStart
            | EconomyStage::InputMarketPrepare
            | EconomyStage::InputMarketRepeat
            | EconomyStage::MarketBoundary => {
                if id == 0 {
                    control[LATCH_WORD].store(control[crate::simulation::flags::FAULT_WORD].load());
                    if comptime!(stage != EconomyStage::MarketBoundary) {
                        control[CONSTRUCTED_EVENTS_WORD].store(0);
                        control[RANGE_SLOTS_WORD].store(0);
                        control[markets::SEARCHES_WORD].store(0);
                        control[search::INPUT_MINIMUM_QUOTE_WORD].store(u32::MAX);
                        if comptime!(matches!(parameters.market, MarketFlag::Input(_))) {
                            control[search::INPUT_RAIL_WORD].store(0);
                        }
                        if comptime!(
                            parameters.market
                                == MarketFlag::Input(crate::simulation::flags::InputMarket::Grain)
                        ) {
                            control[search::GRAIN_FIRMS_WORD].store(0);
                        }
                        if comptime!(
                            parameters.empty_orders
                                == crate::simulation::parameters::EmptyOrderMode::FeedbackOnly
                        ) {
                            control[markets::WIDE_ORDERS_WORD].store(0);
                        }
                    }
                    if comptime!(matches!(
                        stage,
                        EconomyStage::InputMarketStart | EconomyStage::InputMarketPrepare
                    )) {
                        control[markets::PREPARATION_ACTORS_WORD].store(0);
                    }
                }
                if comptime!(matches!(
                    stage,
                    EconomyStage::InputMarketStart | EconomyStage::InputMarketRepeat
                )) && control[crate::simulation::flags::FAULT_WORD].load() == 0
                    && id < parameters.count
                {
                    KernelEventStatus::new(comptime!(EventStatus::Inactive))
                        .store(bundles, 4 * parameters.count + id);
                }
            }
            EconomyStage::ProposeCompact | EconomyStage::ProposeRetainedCompact => {
                let actor = (id + parameters.settlement_rotation) % max(parameters.count, 1);
                let peer = proposals::propose(
                    economy,
                    spatial,
                    proposals,
                    control,
                    parameters,
                    select(id < parameters.count, actor, config.actors),
                    comptime!(match stage {
                        EconomyStage::ProposeRetainedCompact => employment::LaborRound::Retained,
                        _ => employment::LaborRound::Vacancies,
                    }),
                );
                markets::compact_proposal(peer, bundles, parameters, id);
            }
            EconomyStage::GroupCompactFill => {
                if control[crate::simulation::flags::FAULT_WORD].load() == 0 {
                    markets::compact_fill(proposals, bundles, control, parameters, id);
                }
            }
            EconomyStage::GroupRadixLocal => markets::radix_local(bundles, control, parameters, id),
            EconomyStage::GroupRadixScatter => {
                markets::radix_scatter(bundles, control, parameters, id)
            }
            EconomyStage::SupplierRanges => {
                markets::supplier_ranges(bundles, &*spatial, control, config, parameters, id)
            }
            EconomyStage::GroupRanges => {
                if control[crate::simulation::flags::FAULT_WORD].load() == 0 {
                    markets::ranges(bundles, control, config, parameters, id);
                }
            }
            EconomyStage::ClearFeedback => {
                if id < control.len() as u32 {
                    control[id as usize].store(0);
                }
            }
            EconomyStage::RecordFeedback => feedback::record_feedback(
                &mut economy.ledger,
                &*spatial,
                proposals,
                bundles,
                control,
                parameters,
                id,
            ),
            EconomyStage::ApplyFeedback => {
                feedback::apply_feedback(&mut economy.ledger, control, id)
            }
            EconomyStage::IndexCount => {
                indexing::index_count(&*economy, spatial, bundles, parameters, id)
            }
            EconomyStage::IndexOffsets => {
                indexing::index_offsets(config, spatial, bundles, parameters, id)
            }
            EconomyStage::IndexFill => {
                indexing::index_fill(config, spatial, bundles, parameters, id)
            }
            EconomyStage::IndexPrices => indexing::index_prices(&economy.ledger, spatial),
            EconomyStage::FirmRent => rent::collect(economy, control, id, rent::Tenants::Firms),
            EconomyStage::HouseholdRent => {
                rent::collect(economy, control, id, rent::Tenants::Households)
            }
            EconomyStage::Local => phases::local(economy, control, parameters, id),
            EconomyStage::Propose => {
                proposals::propose(
                    economy,
                    spatial,
                    proposals,
                    control,
                    parameters,
                    id,
                    comptime!(employment::LaborRound::FromClock),
                );
            }
            EconomyStage::MarketActivity => {
                let mut active = 0u32;
                if id < config.actors {
                    active = u32::cast_from(
                        proposals[id as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR]
                            != NO_ACTOR,
                    );
                }
                let total = parallel::reduce::<u32>(active, comptime!(Reduction::Sum));
                if UNIT_POS == 0 && total != 0 {
                    control[CONSTRUCTED_EVENTS_WORD].fetch_add(total);
                }
            }
            EconomyStage::Construct => construct(
                economy,
                proposals,
                bundles,
                control,
                parameters,
                id,
                comptime!(EventStorage::Buffer),
            ),
            EconomyStage::GroupInitialize => {
                groups::group_initialize(proposals, bundles, control, parameters, id)
            }
            EconomyStage::GroupSort => groups::group_sort(bundles, control, parameters, id),
            EconomyStage::GroupMerge => groups::group_merge(bundles, control, parameters, id),
            EconomyStage::SettleInputTile => {
                groups::settle_input_tile(economy, proposals, bundles, control, parameters)
            }
            EconomyStage::GroupExecute => {
                groups::group_execute(economy, proposals, bundles, control, parameters, id)
            }
            EconomyStage::GroupRemainder => {
                groups::group_remainder(economy, proposals, bundles, control, parameters, id)
            }
            EconomyStage::GroupPrepare => actor_settlement(
                economy,
                proposals,
                bundles,
                control,
                parameters,
                id,
                comptime!(Posting::CheckActor),
            ),
            EconomyStage::GroupFinalize | EconomyStage::FinalizeFixedWages => {
                if comptime!(stage == EconomyStage::FinalizeFixedWages)
                    && id < markets::active(control, parameters)
                {
                    let count = parameters.count;
                    let peer = bundles[(parameters.sort_source + id) as usize];
                    if id >= bundles[(7 * count + peer) as usize] {
                        let actor = bundles[(parameters.sort_source + count + id) as usize];
                        let prior = KernelEventStatus::load(bundles, 4 * count + actor);
                        let mut status = KernelEventStatus::new(comptime!(EventStatus::Rejected));
                        if prior.contains(comptime!(EventStatus::Faulted)) {
                            status.set(comptime!(EventStatus::Faulted));
                        }
                        status.store(bundles, 4 * count + actor);
                        proposals[(actor + parameters.offset) as usize * ProposalLayout::WORDS
                            + ProposalLayout::QUANTITY_HIGH] = 0;
                    }
                }
                actor_settlement(
                    economy,
                    proposals,
                    bundles,
                    control,
                    parameters,
                    id,
                    comptime!(Posting::ApplyActor),
                );
            }
            EconomyStage::Summarize => observation::summarize(&*economy, bundles, control, id),
            EconomyStage::Inspect => {
                observation::inspect(&economy.ledger, proposals, parameters, id)
            }
        }
    }
}

pub(super) mod accounting;
pub(super) mod households;
pub(super) mod market;
pub(super) mod phases;
pub(super) mod production;
pub(super) mod property;
pub(super) mod settlement;
mod systems;
pub(super) mod transport;
use accounting::observation;
use households::{consumption, employment, food, production as household_production};
use market::{feedback, indexing, orders, pricing, proposals, search, sorting};
use production::{enterprises, farming};
use property::{landlords, rent};
use settlement::{batch, clearing, events, groups, scheduler};
use transport::{cartage, railway};

use super::{
    flags::{
        ActorKind, COMPLETED_WORD, CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, EventStatus, FaultFlags,
        InputMarket, KernelActorKind, KernelAuthorization, KernelEventStatus, KernelFaults,
        MarketFlag, PROGRESS_WORD,
    },
    households::{KernelFatigue, KernelObservation, ObservationKind},
    layout::*,
    negotiation::{NegotiationRequest, NegotiationStatus, legal_grid_rate, negotiate},
    parallel::{
        self, Reduction,
        workgroup::{self, TickTree, WorkgroupTree},
    },
    parameters::{Discovery, EconomyParameters},
    pass::GpuPass,
    payroll::PayrollAccrual,
    policy::*,
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
use clearing::{PeerBalances, PeerOffer, actor_settlement};
use cubecl::prelude::*;
use events::{EventStorage, construct};
use market::demand::{PurchaseRequest, trial_packet};
use market::orders::{LATCH_WORD, RANGE_SLOTS_WORD};
use proposals::MarketProposal;
use settlement::scheduler::{Posting, apply};
pub(super) use systems::{EconomyInputLaunch, EconomySystem};

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

impl EconomyStage {
    fn requires_open_market(self) -> bool {
        matches!(
            self,
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
    }
}

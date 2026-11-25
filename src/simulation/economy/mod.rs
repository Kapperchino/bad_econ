pub(super) mod accounting;
pub(super) mod configuration;
pub(super) mod households;
pub(super) mod market;
pub(super) mod parameters;
pub(super) mod phases;
pub(super) mod policy;
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
    households::{KernelFatigue, KernelObservation, ObservationKind},
    negotiation::{NegotiationRequest, NegotiationStatus, legal_grid_rate, negotiate},
    parallel::{
        self, Reduction,
        workgroup::{self, TickTree, WorkgroupTree},
    },
    pass::GpuPass,
    storage::{
        flags::{
            ActorKind, COMPLETED_WORD, CONSTRUCTED_EVENTS_WORD, CONTROL_WORDS, ConsumerMarket,
            EventStatus, FaultFlags, InputMarket, KernelActorKind, KernelAuthorization,
            KernelEventStatus, KernelFaults, MarketFlag, PROGRESS_WORD,
        },
        layout::*,
        state::*,
        words::*,
    },
    system::{AsBufferArg, System, gpu_passes},
};
use crate::{
    domain::{
        BASE_FIELDS, CONSUMPTION_FIELDS, Field, Good, PRODUCTION_FIELDS, Phase, TRANSIT_FIELDS,
        USED_FIELDS,
    },
    ledger::{AuthorizationFlags, EMPTY, LEG_WORDS, LegLayout, MAX_LEGS},
};
use clearing::{PeerBalances, PeerOffer, actor_settlement};
use configuration::{EconomyConfig, Feature};
use cubecl::prelude::*;
use events::{EventStorage, construct};
use market::demand::{PurchaseRequest, trial_packet};
use market::orders::{LATCH_WORD, RANGE_SLOTS_WORD};
use parameters::{Discovery, EconomyParameters};
use policy::*;
use proposals::MarketProposal;
use settlement::payroll::PayrollAccrual;
use settlement::scheduler::{Posting, apply};
pub(super) use systems::{EconomyInputLaunch, EconomySystem, feedback_blocks, settle_blocks};

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
    RetailOrders,
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

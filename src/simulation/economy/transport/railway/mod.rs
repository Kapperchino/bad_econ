use crate::domain::railway::{RAIL_ROUTE_WORDS as ROUTE_WORDS, RailCapacityField, RouteStorage};
use crate::simulation::DeviceEconomy;
use crate::simulation::economy::*;
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

#[cfg(test)]
mod tests;

pub(in crate::simulation) mod settlement;

const COMPANY_WORDS: usize = 4;
const SEGMENT_WORDS: usize = 7;
const REGION_WORDS: usize = 3;
const UNREACHABLE: u32 = u32::MAX / 4;

mod network;
mod operations;
mod regional;
mod route_table;
mod routing;

use network::{RailNetworkExpand, RouteCacheState};
use operations::{close, dormant_demand, infrastructure, record_use};
use regional::RegionalCapacity;

pub(in crate::simulation) use network::RouteAvailability;
pub use network::{Network, RailNetwork};
pub(in crate::simulation::economy) use operations::{demand, offer_crew_jobs};
pub(in crate::simulation::economy) use routing::{DeliveryRequest, FreightRequest, FreightRoute};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum RailStage {
    Infrastructure,
    DormantDemand,
    Access,
    Routes,
    Quotes,
    Close,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RailPass {
    Infrastructure,
    DormantDemand,
    Access,
    DirtyCarriers,
    PrepareRoutes,
    Routes,
    Quotes,
    Inbound,
    CompleteQuotes,
    Close,
}

#[derive(CubeType, CubeLaunch)]
struct RailInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    stage: RailPass,
}

struct RailSystem;
struct AdvanceRailways;
impl System for RailSystem {
    type Input = RailInput;
    type Passes = gpu_passes![AdvanceRailways];
}

#[cube]
impl GpuPass<RailInput> for AdvanceRailways {
    type Output = ();
    fn execute(mut input: RailInput) {
        let levels = LedgerColumn::new(&input.state, &input.config, Field::RailOperating);
        match comptime!(input.stage) {
            RailPass::Infrastructure => infrastructure(
                &mut EconomyView::new(&mut input.state, &input.actors, &input.config),
                &input.control,
            ),
            RailPass::DormantDemand => {
                dormant_demand(&mut Ledger::new(&mut input.state, &input.config))
            }
            RailPass::Access => input
                .config
                .railway
                .refresh_access(&ActorTable::new(&input.actors), input.config.actors),
            RailPass::DirtyCarriers => input.config.railway.mark_dirty_carriers(&levels),
            RailPass::PrepareRoutes => input.config.railway.prepare_routes(&levels),
            RailPass::Routes => input.config.railway.rebuild_routes(&levels),
            RailPass::Quotes => input
                .config
                .railway
                .refresh_quotes(input.config.ring, input.config.shipment_speed),
            RailPass::Inbound => input.config.railway.refresh_inbound(input.config.ring),
            RailPass::CompleteQuotes => input.config.railway.complete_quotes(),
            RailPass::Close => close(
                &mut Ledger::new(&mut input.state, &input.config),
                &input.control,
            ),
        }
    }
}

pub fn launch(
    DeviceEconomy {
        device,
        state,
        actors,
        config,
    }: DeviceEconomy<'_>,
    control: &CubeTensor,
    stage: RailStage,
) {
    if config.railway.company_count > 0 {
        let passes: &[RailPass] = match stage {
            RailStage::Infrastructure => &[RailPass::Infrastructure],
            RailStage::DormantDemand => &[RailPass::DormantDemand],
            RailStage::Access => &[RailPass::Access],
            RailStage::Routes => &[
                RailPass::DirtyCarriers,
                RailPass::PrepareRoutes,
                RailPass::Routes,
            ],
            RailStage::Quotes if config.railway.route_index != 0 => &[
                RailPass::Quotes,
                RailPass::Inbound,
                RailPass::CompleteQuotes,
            ],
            RailStage::Quotes => &[RailPass::Quotes, RailPass::CompleteQuotes],
            RailStage::Close => &[RailPass::Close],
        };
        for &stage in passes {
            let count = match stage {
                RailPass::Infrastructure | RailPass::DirtyCarriers => {
                    config.railway.company_count * device.workgroup
                }
                RailPass::Close => config.railway.company_count,
                RailPass::DormantDemand => config.railway.region_count,
                RailPass::Access => config.actors,
                RailPass::Routes => config.railway.route_rows,
                RailPass::Inbound => config.railway.nodes,
                RailPass::Quotes => config.railway.nodes * config.railway.nodes * config.ring,
                RailPass::PrepareRoutes | RailPass::CompleteQuotes => 1,
            };
            RailSystem::launch_elements(
                device,
                count as usize,
                RailInputLaunch {
                    state: state.buffer_arg(),
                    actors: actors.buffer_arg(),
                    config: config.launch(),
                    control: control.buffer_arg(),
                    stage,
                },
            );
        }
    }
}

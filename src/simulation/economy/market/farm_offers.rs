use crate::domain::{ActorRows, Market, enterprise::EnterpriseState};
use crate::simulation::economy::*;
use crate::simulation::{Device, DeviceEconomy};
use burn_wgpu::CubeTensor;

pub(in crate::simulation) const MAILBOX_START: usize = Good::COUNT + 5;

pub(in crate::simulation) struct FarmMarket {
    firms: CubeTensor,
    bids: CubeTensor,
    count: usize,
}

impl FarmMarket {
    pub fn new(device: &Device, actors: &ActorRows) -> Self {
        let mut firms = actors.firm_indices().collect::<Vec<_>>();
        firms.sort_unstable_by_key(|&firm| (actors.at(firm).position.x, firm));
        let count = firms.len();
        let directory = firms
            .into_iter()
            .map(|firm| firm as u32)
            .collect::<Vec<_>>();
        Self {
            firms: device.record_tensor(&directory),
            bids: device.empty(3 * count + 1),
            count,
        }
    }

    pub fn takes_offers(market: Market, home: Feature) -> bool {
        matches!(
            (market, home),
            (Market::Input(Good::Grain), _) | (Market::Input(Good::Bread), Feature::Enabled)
        )
    }

    pub fn collect(
        &self,
        gpu: DeviceEconomy<'_>,
        spatial: &CubeTensor,
        control: &CubeTensor,
        parameters: &EconomyParameters,
    ) {
        if self.count > 0 {
            let device = gpu.device;
            let dim = CubeDim::new_1d(device.workgroup);
            publish_bids::launch(
                &device.client,
                device.count(self.count),
                dim,
                gpu.state.buffer_arg(),
                gpu.actors.buffer_arg(),
                gpu.config.launch(),
                spatial.buffer_arg(),
                control.buffer_arg(),
                self.firms.buffer_arg(),
                self.bids.buffer_arg(),
                gpu.config.market_table.buffer_arg(),
                parameters.launch(),
            );
            parallel::scan(device, &self.bids, 0, self.count);
            index_bids::launch(
                &device.client,
                device.count(self.count),
                dim,
                self.bids.buffer_arg(),
                self.count as u32,
            );
            submit_surplus::launch(
                &device.client,
                device.count(gpu.config.actors as usize),
                dim,
                gpu.state.buffer_arg(),
                gpu.actors.buffer_arg(),
                gpu.config.launch(),
                self.firms.buffer_arg(),
                self.bids.buffer_arg(),
                gpu.config.market_table.buffer_arg(),
                parameters.launch(),
            );
        }
    }
}

#[cube(launch)]
fn publish_bids(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    control: &[Atomic<u32>],
    firms: &[u32],
    bids: &mut [u32],
    inbox: &mut [u32],
    parameters: &EconomyParameters,
) {
    let slot = ABSOLUTE_POS;
    if slot < firms.len() {
        let firm = firms[slot];
        let mut economy = EconomyView::new(state, actors, config);
        let mut budget = 0u32;
        if pricing::distributor(&economy.ledger, firm)
            && economy.ledger.number(Field::EnterpriseStatus, firm)
                == comptime!(EnterpriseState::Operating as u32)
        {
            let order = PurchaseRequest::new(&mut economy, spatial, control, parameters, firm);
            budget = select(
                order.search,
                minimum(order.purchasing_power, ticks(u32::MAX, 0)).extract(LOW_WORD),
                0,
            );
        }
        bids[slot] = u32::cast_from(budget > 0);
        bids[2 * firms.len() + slot] = budget;
        let mailbox = comptime!(MAILBOX_START) + firm as usize;
        inbox[mailbox] = 0;
        inbox[mailbox + config.actors as usize] = 0;
    }
}

#[cube(launch)]
fn index_bids(bids: &mut [u32], count: u32) {
    let slot = ABSOLUTE_POS as u32;
    if slot < count {
        let end = bids[slot as usize];
        let start = select(slot > 0, bids[(max(slot, 1) - 1) as usize], 0);
        if end > start {
            bids[(count + start) as usize] = slot;
        }
    }
}

#[derive(CubeType)]
struct FarmOffer {
    seller: u32,
    price: u32,
    rotation: u32,
}

#[derive(CubeType)]
struct DistributorSearch {
    distance: u32,
    priority: u32,
    buyer: u32,
}

#[cube]
impl DistributorSearch {
    fn visit(
        &mut self,
        economy: &EconomyView<'_>,
        firms: &[u32],
        bids: &[u32],
        index: u32,
        offer: &FarmOffer,
    ) -> bool {
        let n = firms.len();
        let slot = bids[n + index as usize] as usize;
        let buyer = firms[slot];
        let config = economy.ledger.config;
        let x = economy.actors.x(offer.seller);
        let bx = economy.actors.x(buyer);
        let within = max(x, bx) - min(x, bx) <= self.distance;
        let distance = economy.actors.distance(buyer, offer.seller);
        let priority = buyer ^ offer.rotation ^ offer.seller;
        if distance < self.distance || (distance == self.distance && priority < self.priority) {
            let mut access = 0u32;
            if config.railway.has_routes() {
                access = config.railway.access[buyer as usize * 2 + 1];
            }
            let days = railway::DeliveryRequest {
                buyer,
                seller: offer.seller,
                trip: distance,
                buyer_access: access,
            }
            .days(config);
            let delivered = cartage::delivered_price(config, offer.price, distance, days);
            if delivered <= bids[2 * n + slot] {
                self.buyer = buyer;
                self.distance = distance;
                self.priority = priority;
            }
        }
        within
    }
}

#[cube]
impl FarmOffer {
    fn nearest(&self, economy: &EconomyView<'_>, firms: &[u32], bids: &[u32], count: u32) -> u32 {
        let x = economy.actors.x(self.seller);
        let n = firms.len();
        let mut left = 0u32;
        let mut right = count;
        while left < right {
            let mid = left + (right - left) / 2;
            if economy.actors.x(firms[bids[n + mid as usize] as usize]) < x {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        let mut search = DistributorSearch {
            distance: economy.ledger.config.search_radius,
            priority: u32::MAX,
            buyer: NO_ACTOR,
        };
        let mut visit_left = left > 0;
        let mut visit_right = right < count;
        while visit_left || visit_right {
            if visit_left {
                left -= 1;
                visit_left = search.visit(economy, firms, bids, left, self) && left > 0;
            }
            if visit_right {
                let within = search.visit(economy, firms, bids, right, self);
                right += 1;
                visit_right = within && right < count;
            }
        }
        search.buyer
    }
}

#[cube(launch)]
fn submit_surplus(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    firms: &[u32],
    bids: &[u32],
    inbox: &[Atomic<u32>],
    parameters: &EconomyParameters,
) {
    let seller = ABSOLUTE_POS as u32;
    let n = firms.len();
    let count = bids[n - 1];
    if seller < config.actors && count > 0 {
        let economy = EconomyView::new(state, actors, config);
        let output = economy.actors.kind(seller).output();
        let good = parameters.traded_good();
        let household = output == 0 && economy.ledger.number(Field::LandPlots, seller) > 0;
        let farm = good == comptime!(Good::Grain as u32)
            && output == good
            && !pricing::distributor(&economy.ledger, seller);
        if (household || farm) && household_production::offers(&economy, seller, good) {
            let terms =
                household_production::TradeFields::new(&economy.actors, config, seller, good);
            let price = economy.ledger.number_at(terms.price, seller);
            let offer = FarmOffer {
                seller,
                price,
                rotation: parameters.settlement_rotation,
            };
            let buyer = offer.nearest(&economy, firms, bids, count);
            if buyer != NO_ACTOR && price > 0 {
                let meals = times32(
                    economy.ledger.number(Field::NutritionNeed, seller),
                    config.coverage + config.ring,
                );
                let funds = positive(
                    economy.ledger.value(Field::Cash, seller),
                    economy.ledger.value(Field::CartagePayable, seller),
                );
                let urgent = household
                    && good == comptime!(Good::Grain as u32)
                    && lt(
                        funds,
                        times(meals, economy.ledger.number(Field::FoodPrice, seller)).value,
                    );
                let mailbox = comptime!(MAILBOX_START)
                    + buyer as usize
                    + u32::cast_from(urgent) as usize * config.actors as usize;
                inbox[mailbox].fetch_max(u32::MAX - (seller ^ offer.rotation));
            }
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn sender(
    config: &EconomyConfig,
    buyer: u32,
    rotation: u32,
) -> u32 {
    let mailbox = comptime!(MAILBOX_START) + buyer as usize;
    let urgent = config.market_table[mailbox + config.actors as usize];
    let offer = select(urgent > 0, urgent, config.market_table[mailbox]);
    select(offer == 0, NO_ACTOR, (u32::MAX - offer) ^ rotation)
}

use crate::domain::{ActorRows, Market, enterprise::EnterpriseState};
use crate::simulation::economy::*;
use crate::simulation::{Device, DeviceEconomy};
use burn_wgpu::CubeTensor;

pub(in crate::simulation) const MAILBOX_START: usize = Good::COUNT + 5;

const BID_ROWS: u32 = 32;
const BID_WORDS: usize = 4;
const BID_COLUMNS: usize = 3 + BID_WORDS;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::simulation) enum BidRound {
    Open,
    Continue,
}

pub(in crate::simulation) struct FarmMarket {
    firms: CubeTensor,
    bids: CubeTensor,
    count: usize,
}

impl FarmMarket {
    pub fn new(device: &Device, actors: &ActorRows, map_size: u32) -> Self {
        let height = map_size.div_ceil(BID_ROWS);
        let mut firms = actors.firm_indices().collect::<Vec<_>>();
        firms.sort_by_cached_key(|&firm| {
            let position = actors.at(firm).position;
            (position.y / height, position.x, firm)
        });
        let count = firms.len();
        let mut bids = vec![0; BID_COLUMNS * count + BID_ROWS as usize + 1];
        let borders = &mut bids[BID_COLUMNS * count..];
        for &firm in &firms {
            let row = actors.at(firm).position.y / height;
            borders[row as usize + 1] += 1;
        }
        for row in 1..borders.len() {
            borders[row] += borders[row - 1];
        }
        let directory = firms
            .into_iter()
            .map(|firm| firm as u32)
            .collect::<Vec<_>>();
        Self {
            firms: device.record_tensor(&directory),
            bids: device.tensor(&bids),
            count,
        }
    }

    pub fn graph_buffers(&self) -> [&CubeTensor; 2] {
        [&self.firms, &self.bids]
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
        round: BidRound,
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
                round,
            );
            parallel::scan(device, &self.bids, 0, self.count);
            index_bids::launch(
                &device.client,
                device.count(self.count),
                dim,
                gpu.actors.buffer_arg(),
                self.firms.buffer_arg(),
                self.bids.buffer_arg(),
            );
            submit_surplus::launch(
                &device.client,
                device.count(gpu.config.actors as usize),
                dim,
                gpu.state.buffer_arg(),
                gpu.actors.buffer_arg(),
                gpu.config.launch(),
                self.count as u32,
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
    mut config: EconomyConfig,
    spatial: &[u32],
    control: &[Atomic<u32>],
    firms: &[u32],
    bids: &mut [u32],
    inbox: &mut [u32],
    mut parameters: EconomyParameters,
    #[comptime] round: BidRound,
) {
    parameters.refresh(&mut config);
    let slot = ABSOLUTE_POS;
    if slot < firms.len() {
        let firm = firms[slot];
        let mut economy = EconomyView::new(state, actors, &config);
        let mut budget = 0u32;
        let open = comptime!(round == BidRound::Open)
            || control[orders::PREPARATION_ACTORS_WORD].load() > 0;
        if open
            && pricing::distributor(&economy.ledger, firm)
            && economy.ledger.number(Field::EnterpriseStatus, firm)
                == comptime!(EnterpriseState::Operating as u32)
        {
            let order = PurchaseRequest::new(
                &mut economy,
                spatial,
                control,
                &parameters,
                firm,
                comptime!(pricing::QuoteSource::FromLedger),
            );
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
fn index_bids(actors: &[u32], firms: &[u32], bids: &mut [u32]) {
    let count = firms.len() as u32;
    let slot = ABSOLUTE_POS as u32;
    if slot < count {
        let end = bids[slot as usize];
        let start = select(slot > 0, bids[(max(slot, 1) - 1) as usize], 0);
        if end > start {
            let buyers = ActorTable::new(actors);
            let buyer = firms[slot as usize];
            let record = 3 * count as usize + start as usize * comptime!(BID_WORDS);
            bids[record] = buyer;
            bids[record + 1] = buyers.x(buyer);
            bids[record + 2] = buyers.y(buyer);
            bids[record + 3] = bids[(2 * count + slot) as usize];
        }
    }
}

#[derive(CubeType)]
struct FarmOffer {
    seller: u32,
    x: u32,
    y: u32,
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
        firms: u32,
        bids: &[u32],
        index: u32,
        offer: &FarmOffer,
    ) -> bool {
        let record = 3 * firms as usize + index as usize * comptime!(BID_WORDS);
        let buyer = bids[record];
        let bx = bids[record + 1];
        let by = bids[record + 2];
        let config = economy.ledger.config;
        let dx = max(offer.x, bx) - min(offer.x, bx);
        let within = dx <= self.distance;
        let distance = dx + max(offer.y, by) - min(offer.y, by);
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
            if delivered <= bids[record + 3] {
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
    fn visit_row(
        &self,
        search: &mut DistributorSearch,
        economy: &EconomyView<'_>,
        firms: u32,
        bids: &[u32],
        row: u32,
    ) {
        let x = self.x;
        let n = firms as usize;
        let first_slot = bids[comptime!(BID_COLUMNS) * n + row as usize];
        let past_slot = bids[comptime!(BID_COLUMNS) * n + row as usize + 1];
        let first = select(first_slot == 0, 0, bids[(max(first_slot, 1) - 1) as usize]);
        let past = select(past_slot == 0, 0, bids[(max(past_slot, 1) - 1) as usize]);
        let mut left = first;
        let mut right = past;
        while left < right {
            let mid = left + (right - left) / 2;
            if bids[3 * n + mid as usize * comptime!(BID_WORDS) + 1] < x {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        let mut visit_left = left > first;
        let mut visit_right = right < past;
        while visit_left || visit_right {
            if visit_left {
                left -= 1;
                visit_left = search.visit(economy, firms, bids, left, self) && left > first;
            }
            if visit_right {
                let within = search.visit(economy, firms, bids, right, self);
                right += 1;
                visit_right = within && right < past;
            }
        }
    }

    fn nearest(&self, economy: &EconomyView<'_>, firms: u32, bids: &[u32], count: u32) -> u32 {
        let config = economy.ledger.config;
        let mut search = DistributorSearch {
            distance: config.search_radius,
            priority: u32::MAX,
            buyer: NO_ACTOR,
        };
        if count <= 16 {
            for slot in 0..count {
                search.visit(economy, firms, bids, slot, self);
            }
        } else {
            let y = self.y;
            let height = ceildiv(config.map_size, comptime!(BID_ROWS));
            let row = y / height;
            let mut lower = row;
            let mut upper = row;
            self.visit_row(&mut search, economy, firms, bids, row);
            let last = (config.map_size - 1) / height;
            while lower > (max(y, search.distance) - search.distance) / height
                || upper
                    < min(
                        last,
                        (y + min(search.distance, config.map_size - 1 - y)) / height,
                    )
            {
                if lower > (max(y, search.distance) - search.distance) / height {
                    lower -= 1;
                    self.visit_row(&mut search, economy, firms, bids, lower);
                }
                if upper
                    < min(
                        last,
                        (y + min(search.distance, config.map_size - 1 - y)) / height,
                    )
                {
                    upper += 1;
                    self.visit_row(&mut search, economy, firms, bids, upper);
                }
            }
        }
        search.buyer
    }
}

#[cube(launch)]
fn submit_surplus(
    state: &mut [u32],
    actors: &[u32],
    mut config: EconomyConfig,
    firms: u32,
    bids: &[u32],
    inbox: &[Atomic<u32>],
    mut parameters: EconomyParameters,
) {
    parameters.refresh(&mut config);
    let seller = ABSOLUTE_POS as u32;
    let count = bids[(firms - 1) as usize];
    if seller < config.actors && count > 0 {
        let economy = EconomyView::new(state, actors, &config);
        let output = economy.actors.kind(seller).output();
        let good = parameters.traded_good();
        let household = output == 0 && economy.ledger.number(Field::LandPlots, seller) > 0;
        let farm = good == comptime!(Good::Grain as u32)
            && output == good
            && !pricing::distributor(&economy.ledger, seller);
        if (household || farm) && household_production::offers(&economy, seller, good) {
            let terms =
                household_production::TradeFields::new(&economy.actors, &config, seller, good);
            let price = economy.ledger.number_at(terms.price, seller);
            let offer = FarmOffer {
                seller,
                x: economy.actors.x(seller),
                y: economy.actors.y(seller),
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

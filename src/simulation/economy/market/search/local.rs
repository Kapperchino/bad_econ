use super::*;

#[cube]
pub(in crate::simulation::economy) fn staple(
    economy: &EconomyView<'_>,
    spatial: &[u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    buyer: u32,
    wanted: Ticks,
    cash: Ticks,
) -> MarketProposal {
    let config = economy.ledger.config;
    let actors = &economy.actors;
    let good = parameters.traded_good();
    let mut proposal = MarketProposal {
        peer: NO_ACTOR,
        quantity: zero(),
        trip: 0,
    };
    let dealer = pricing::distributor(&economy.ledger, buyer);
    if dealer
        && (good == Good::Grain as u32
            || (good == Good::Bread as u32 && comptime!(config.home_mode == Feature::Enabled)))
    {
        let seller = market::farm_offers::sender(config, buyer, parameters.settlement_rotation);
        if seller < config.actors {
            let trip = actors.distance(buyer, seller);
            let offer = OfferRequest {
                buyer,
                seller,
                trip,
                buyer_access: BuyerLocation::new(actors, config, buyer).access,
                demand: BuyerDemand {
                    wanted,
                    purchasing_power: cash,
                },
            }
            .evaluate(economy, control, parameters);
            if offer.eligible {
                proposal.peer = seller;
                proposal.quantity = offer.quantity;
                proposal.trip = trip;
            }
        }
    }
    if !dealer
        && comptime!(
            matches!(parameters.market, MarketFlag::Input(_))
                && config.home_mode == Feature::Enabled
                && config.economy == crate::domain::Economy::Industrial
        )
        && (good == Good::Grain as u32 || good == Good::Flour as u32)
    {
        let base = config.query_members();
        let width = ceildiv(config.map_size, 128);
        let side = ceildiv(config.map_size, width);
        let tile = actors.y(buyer) / width * side + actors.x(buyer) / width;
        let first = spatial[(base + config.actors + tile) as usize];
        let end = spatial[(base + config.actors + tile + 1) as usize];
        let count = end - first;
        let mut index = 0u32;
        while index < min(count, GRAIN_MARKET_QUOTES) && proposal.peer == NO_ACTOR {
            let slot = end - 1 - index;
            let seller = spatial[(base + slot) as usize];
            if actors.distance(buyer, seller) == 0
                && super::super::indexing::discoverable(economy, parameters, seller)
            {
                let offer = OfferRequest {
                    buyer,
                    seller,
                    trip: 0,
                    buyer_access: BuyerLocation::new(actors, config, buyer).access,
                    demand: BuyerDemand {
                        wanted,
                        purchasing_power: cash,
                    },
                }
                .evaluate(economy, control, parameters);
                if offer.eligible {
                    proposal.peer = seller;
                    proposal.quantity = offer.quantity;
                }
            }
            index += 1;
        }
    }
    proposal
}

#[cube]
impl SearchInput {
    pub(super) fn collect_local(&mut self) {
        let config = &self.config;
        let count = self.control[SEARCHES_WORD].load();
        let mut group = CUBE_POS as u32;
        while group < ceildiv(count, CUBE_DIM) {
            let job = group * CUBE_DIM + UNIT_POS;
            let mut buyer = 0u32;
            let mut matched = 0u32;
            let mut unmatched = 0u32;
            if job < count {
                buyer = self.searches[job as usize];
                let start = buyer as usize * ProposalLayout::WORDS;
                let seller = self.proposals[start + ProposalLayout::ACTOR];
                if seller != NO_ACTOR {
                    matched = 1;
                    let trip = self.proposals[start + ProposalLayout::DISTANCE];
                    if config.railway.has_routes() && self.control[INPUT_RAIL_WORD].load() == 0 {
                        let delivery = railway::DeliveryRequest {
                            buyer,
                            seller,
                            trip,
                            buyer_access: config.railway.access[buyer as usize * 2 + 1],
                        };
                        if delivery.may_book(config) {
                            self.control[INPUT_RAIL_WORD].store(1);
                        }
                    }
                    record_success(&self.successes, &self.parameters, buyer);
                } else {
                    unmatched = u32::cast_from(
                        comptime!(self.parameters.market != MarketFlag::Input(InputMarket::Grain))
                            || !pricing::distributor(&Ledger::new(&mut self.state, config), buyer),
                    );
                }
            }
            let prefix =
                parallel::scan_prefix(unmatched | (matched << 16), comptime!(self.parameters.scan));
            let mut missing = Shared::<u32>::new();
            let mut found = Shared::<u32>::new();
            if UNIT_POS == 0 {
                let total = prefix.total & 0xffff;
                if total > 0 {
                    *missing = self.control[UNMATCHED_WORD].fetch_add(total);
                }
                if comptime!(self.program.collection == OrderCollection::InputTile)
                    && prefix.total >> 16 > 0
                {
                    *found = self.control[CONSTRUCTED_EVENTS_WORD].fetch_add(prefix.total >> 16);
                }
            }
            sync_cube();
            if unmatched != 0 {
                let slot = *missing + (prefix.inclusive & 0xffff) - 1;
                self.searches[(comptime!(UNMATCHED_FIELD) * config.actors + slot) as usize] = buyer;
            }
            if comptime!(self.program.collection == OrderCollection::InputTile) && matched != 0 {
                let slot = *found + (prefix.inclusive >> 16) - 1;
                if slot < comptime!(sorting::INPUT_TILE as u32) {
                    self.searches
                        [(comptime!(INPUT_BUYERS_FIELD) * config.actors + slot) as usize] = buyer;
                }
            }
            sync_cube();
            group += CUBE_COUNT as u32;
        }
    }
}

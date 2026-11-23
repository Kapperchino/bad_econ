use super::*;

#[cube]
fn tile_offset(
    actors: &ActorTable<'_>,
    spatial: &[u32],
    base: u32,
    count: u32,
    tile: u32,
    width: u32,
    side: u32,
) -> u32 {
    let mut first = 0u32;
    let mut end = count;
    while first < end {
        let mid = first + (end - first) / 2;
        let actor = spatial[(base + mid) as usize];
        let cell = actors.y(actor) / width * side + actors.x(actor) / width;
        if cell < tile {
            first = mid + 1;
        } else {
            end = mid;
        }
    }
    first
}

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
        let cells = config.spatial_side * config.spatial_side;
        let base = 3 * (cells + 1 + config.actors) + ceildiv(config.actors, DISCOVERY_BLOCK) + 1;
        let width = ceildiv(config.map_size, 128);
        let side = ceildiv(config.map_size, width);
        let tile = actors.y(buyer) / width * side + actors.x(buyer) / width;
        let first = tile_offset(actors, spatial, base, config.actors, tile, width, side);
        let end = tile_offset(actors, spatial, base, config.actors, tile + 1, width, side);
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

use crate::domain::enterprise::{EnterpriseModel, EnterpriseState};
use crate::simulation::economy::*;

#[cube]
pub(in crate::simulation::economy) fn distributor(ledger: &Ledger<'_>, firm: u32) -> bool {
    ledger.number(Field::EnterpriseModel, firm) == comptime!(EnterpriseModel::Distributor as u32)
}

#[cube]
pub(in crate::simulation::economy) fn retail_field(
    config: &EconomyConfig,
    good: u32,
    #[comptime] account: crate::domain::RetailAccount,
) -> u32 {
    config.fields - comptime!(crate::domain::RetailAccount::FIELDS as u32)
        + comptime!(account as u32 * Good::COUNT as u32)
        + good
        - 1
}

#[cube]
pub(in crate::simulation::economy) fn cost_field(
    config: &EconomyConfig,
    output: u32,
    good: u32,
) -> u32 {
    select(
        output == good,
        comptime!(Field::ReplacementCost as u32),
        retail_field(config, good, comptime!(crate::domain::RetailAccount::Cost)),
    )
}

#[repr(u32)]
enum InventoryPolicy {
    Producer,
    Distributor,
    Seasonal,
}

#[repr(u32)]
enum QuoteRule {
    Inventory,
    DiscountToCost,
    CostPlusMargin,
    Scarcity,
}

#[repr(u32)]
enum FinishedStockPolicy {
    Market,
    FirstOffer,
    Clearance,
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct PriceQuote {
    actor: u32,
    price: Ticks,
    precise: Ticks,
}

#[cube]
impl PriceQuote {
    pub(in crate::simulation::economy) fn load(ledger: &Ledger<'_>, actor: u32) -> Self {
        PriceQuote {
            actor,
            price: ledger.value(Field::Price, actor),
            precise: ledger.value(Field::PriceQuote, actor),
        }
    }

    pub(in crate::simulation::economy) fn adjust(
        &self,
        ledger: &mut Ledger<'_>,
        control: &[Atomic<u32>],
        multiplier: u32,
    ) {
        let actor = self.actor;
        let config = ledger.config;
        let price = self.price;
        let previous = self.precise;
        let rounded = divide(
            plus(previous, ticks(PRICE_SCALE / 2, 0)).record(control),
            PRICE_SCALE,
        );
        let quote = select(
            equal(rounded, price),
            previous,
            times(price, PRICE_SCALE).record(control),
        );
        let next = divide(times(quote, multiplier).record(control), PRICE_SCALE);
        let next = minimum(
            maximum(next, times32(config.minimum_price, PRICE_SCALE)),
            times32(config.maximum_price, PRICE_SCALE),
        );
        ledger.set(Field::PriceQuote, actor, next);
        ledger.set(
            Field::Price,
            actor,
            divide(
                plus(next, ticks(PRICE_SCALE / 2, 0)).record(control),
                PRICE_SCALE,
            ),
        );
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct ProductOrders {
    pub requested: Ticks,
    pub unfilled: Ticks,
}

#[cube]
impl ProductOrders {
    pub fn load(ledger: &Ledger<'_>, control: &[Atomic<u32>], firm: u32, primary: u32) -> Self {
        let config = ledger.config;
        let mut unfilled = ledger.value(Field::Unfilled, firm);
        let mut requested = plus(ledger.value(Field::Sales, firm), unfilled).record(control);
        if distributor(ledger, firm) {
            requested = ledger.value_at(
                retail_field(
                    config,
                    primary,
                    comptime!(crate::domain::RetailAccount::Orders),
                ),
                firm,
            );
            unfilled = ledger.value(Field::PrimaryRetailUnfilled, firm);
        }
        ProductOrders {
            requested,
            unfilled,
        }
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct FirmPricing<'a> {
    firm: u32,
    good: u32,
    control: &'a [Atomic<u32>],
}

#[cube]
impl<'a> FirmPricing<'a> {
    pub(in crate::simulation::economy) fn new(
        firm: u32,
        good: u32,
        control: &'a [Atomic<u32>],
    ) -> Self {
        FirmPricing::<'a> {
            firm,
            good,
            control,
        }
    }

    fn packet_cost(&self, ledger: &Ledger<'_>) -> Ticks {
        let firm = self.firm;
        let good = self.good;
        let control = self.control;
        let config = ledger.config;
        let payroll = times32(
            ledger.number(Field::Wage, firm),
            ledger.number(Field::LaborMinutes, firm),
        );
        let mut cost = select(
            farming::FarmSite::new(firm, good).is_farm(ledger) && config.home.harvest_days < 365,
            divide(
                times(payroll, farming::CropCalendar::new(config).paid_days(good)).record(control),
                365,
            ),
            payroll,
        );
        for input in 0..2u32 {
            if ledger.number_at(comptime!(Field::FirstInput as u32) + input, firm) > 0 {
                cost = plus(
                    cost,
                    times32(
                        ledger.number(Field::InputTicks, firm),
                        ledger.number_at(comptime!(Field::FirstInputCost as u32) + input, firm),
                    ),
                )
                .record(control);
            }
        }
        if ledger.number(Field::EnterpriseModel, firm)
            == comptime!(EnterpriseModel::Distributor as u32)
        {
            cost = plus(
                cost,
                times32(
                    ledger.number(Field::OutputTicks, firm),
                    ledger.number(Field::ReplacementCost, firm),
                ),
            )
            .record(control);
        }
        cost
    }

    pub(in crate::simulation::economy) fn covers_batch(&self, ledger: &Ledger<'_>) -> bool {
        let firm = self.firm;
        let revenue = times32(
            ledger.number(Field::Price, firm),
            ledger.number(Field::OutputTicks, firm),
        );
        !lt(revenue, self.packet_cost(ledger))
    }

    fn quote_rule(&self, ledger: &Ledger<'_>, excess: Ticks, orders: &ProductOrders) -> u32 {
        let firm = self.firm;
        if equal(excess, zero()) {
            let fulfilled = equal(orders.unfilled, zero())
                && (self.good <= comptime!(Good::Bread as u32) || equal(orders.requested, zero()));
            select(
                fulfilled,
                comptime!(QuoteRule::CostPlusMargin as u32),
                comptime!(QuoteRule::Scarcity as u32),
            )
        } else {
            let dealer = distributor(ledger, firm);
            let stale_retail = dealer
                && equal(orders.requested, zero())
                && lt(
                    ledger.value(Field::Demand, firm),
                    ledger.value(Field::OutputTicks, firm),
                );
            let operating = (ledger.number(Field::FirstInput, firm) > 0 || dealer)
                && !stale_retail
                && ledger.number(Field::EnterpriseStatus, firm)
                    == comptime!(EnterpriseState::Operating as u32);
            select(
                operating,
                comptime!(QuoteRule::DiscountToCost as u32),
                comptime!(QuoteRule::Inventory as u32),
            )
        }
    }

    fn response(
        &self,
        ledger: &Ledger<'_>,
        excess: Ticks,
        orders: &ProductOrders,
        inventory_response: u32,
    ) -> u32 {
        let firm = self.firm;
        let control = self.control;
        let config = ledger.config;
        let mut response = inventory_response;
        let repricing = self.quote_rule(ledger, excess, orders);
        if comptime!(config.economy == crate::domain::Economy::Industrial)
            && comptime!(config.home_mode == Feature::Enabled)
            && config.price_response > 0
            && repricing != comptime!(QuoteRule::Inventory as u32)
        {
            let cost = self.packet_cost(ledger);
            let labor = times32(
                ledger.number(Field::Wage, firm),
                ledger.number(Field::LaborMinutes, firm),
            );
            let margin_base = select(
                self.good <= comptime!(Good::Bread as u32)
                    && (ledger.number(Field::FirstInput, firm) > 0
                        || ledger.number(Field::EnterpriseModel, firm)
                            == comptime!(EnterpriseModel::Distributor as u32)),
                labor,
                cost,
            );
            let target = select(
                repricing == comptime!(QuoteRule::DiscountToCost as u32),
                cost,
                plus(cost, divide(margin_base, 4)).record(control),
            );
            let output = max(ledger.number(Field::OutputTicks, firm), 1);
            let quote = minimum(
                divide(plus(target, ticks(output - 1, 0)).record(control), output),
                ticks(config.maximum_price, 0),
            );
            let price = max(ledger.number(Field::Price, firm), 1);
            let next = plus(
                times32(price, POLICY_SCALE - config.price_response),
                times(quote, config.price_response).record(control),
            )
            .record(control);
            let correction = minimum(
                divide(
                    times(next, PRICE_SCALE / POLICY_SCALE).record(control),
                    price,
                ),
                ticks(u32::MAX, 0),
            )
            .extract(LOW_WORD);
            response = match repricing {
                1 => max(
                    inventory_response,
                    select(
                        equal(orders.requested, zero()),
                        min(correction, PRICE_SCALE),
                        correction,
                    ),
                ),
                2 => correction,
                _ => max(inventory_response, correction),
            };
        }
        response
    }

    fn finished_stock_policy(&self, ledger: &Ledger<'_>, orders: &ProductOrders) -> u32 {
        let firm = self.firm;
        let stock = ledger.value_at(self.good, firm);
        if comptime!(ledger.config.economy == crate::domain::Economy::Industrial)
            && ledger.number(Field::FirstInput, firm) > 0
            && !equal(stock, zero())
            && equal(orders.requested, zero())
            && lt(
                ledger.value(Field::Demand, firm),
                ledger.value(Field::OutputTicks, firm),
            )
        {
            select(
                equal(stock, ledger.value(Field::FinishedToday, firm)),
                comptime!(FinishedStockPolicy::FirstOffer as u32),
                comptime!(FinishedStockPolicy::Clearance as u32),
            )
        } else {
            u32::new(comptime!(FinishedStockPolicy::Market as i64))
        }
    }

    pub(in crate::simulation::economy) fn reprice(
        &self,
        ledger: &mut Ledger<'_>,
        excess: Ticks,
        orders: &ProductOrders,
        inventory_response: u32,
    ) {
        let config = ledger.config;
        let response = match self.finished_stock_policy(&*ledger, orders) {
            1 => u32::new(comptime!(i64::from(PRICE_SCALE))),
            2 => {
                PRICE_SCALE
                    - (PRICE_SCALE - config.price_table[0usize]) * config.price_response
                        / POLICY_SCALE
            }
            _ => self.response(&*ledger, excess, orders, inventory_response),
        };
        let quote = PriceQuote::load(&*ledger, self.firm);
        quote.adjust(ledger, self.control, response);
    }

    pub(in crate::simulation::economy) fn distributor_target(
        &self,
        economy: &EconomyView<'_>,
        demand: Ticks,
        day: u32,
    ) -> Ticks {
        let actor = self.firm;
        let control = self.control;
        let config = economy.ledger.config;
        let reserve = economy.ledger.number(Field::StockCoverage, actor);
        if self.good == comptime!(Good::Grain as u32) {
            let shifts = economy.actors.capacity(actor);
            let units = economy.ledger.number(Field::OutputTicks, actor);
            let capacity = times32(shifts, units);
            let days = min(
                reserve,
                farming::CropCalendar::new(config).on(day).days_to_harvest(),
            );
            let daily = minimum(capacity, maximum(demand, ticks(units, 0)));
            times(daily, days + config.ring).record(control)
        } else {
            times(demand, reserve + config.ring).record(control)
        }
    }

    pub(in crate::simulation::economy) fn inventory_target(
        &self,
        economy: &EconomyView<'_>,
        demand: Ticks,
        day: u32,
    ) -> Ticks {
        let firm = self.firm;
        let good = self.good;
        let control = self.control;
        let config = economy.ledger.config;
        let policy = select(
            farming::FarmSite::new(firm, good).is_farm(&economy.ledger),
            comptime!(InventoryPolicy::Seasonal as u32),
            select(
                economy.ledger.number(Field::EnterpriseModel, firm)
                    == comptime!(EnterpriseModel::Distributor as u32),
                comptime!(InventoryPolicy::Distributor as u32),
                comptime!(InventoryPolicy::Producer as u32),
            ),
        );
        match policy {
            0 => {
                let daily = select(
                    good <= comptime!(Good::Bread as u32),
                    farming::production_target(control, demand),
                    demand,
                );
                let batch = economy.ledger.number(Field::OutputTicks, firm);
                plus(
                    times(daily, config.coverage).record(control),
                    ticks(max(batch, 1) - 1, 0),
                )
                .record(control)
            }
            1 => self.distributor_target(economy, demand, day),
            _ => farming::CropCalendar::new(config)
                .on(day)
                .inventory_target(control, demand),
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn stocked<L: ReadLedger>(
    ledger: &L,
    actor: u32,
    good: u32,
) -> bool {
    good == comptime!(Good::Housing as u32)
        || good == comptime!(Good::Railway as u32)
        || !equal(ledger.value_at(good, actor), zero())
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(in crate::simulation) enum QuoteSource {
    FromLedger,
    Indexed,
    Masked,
}

#[cube]
pub(in crate::simulation::economy) fn offer_price(
    economy: &EconomyView<'_>,
    seller: u32,
    good: u32,
) -> u32 {
    let status = economy.ledger.number(Field::EnterpriseStatus, seller);
    let offered = if good == comptime!(Good::Construction as u32) {
        status == comptime!(EnterpriseState::Operating as u32)
    } else {
        household_production::offers(economy, seller, good)
    };
    let available = (economy.actors.kind(seller).output() == good
        || distributor(&economy.ledger, seller))
        && status < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32)
        && offered;
    if available {
        let terms = household_production::TradeFields::new(
            &economy.actors,
            economy.ledger.config,
            seller,
            good,
        );
        economy.ledger.number_at(terms.price, seller)
    } else {
        0u32
    }
}

pub(in crate::simulation) const QUOTE_MASK_WORDS: u32 = HOUSEHOLD_SEARCH_SAMPLES.div_ceil(32);

#[derive(CubeType)]
pub(in crate::simulation::economy) struct FirmQuotes {
    offers: u32,
    masks: u32,
}

#[cube]
impl FirmQuotes {
    pub(in crate::simulation::economy) fn new(cells: u32, firms: u32) -> Self {
        let offers = cells + 1 + firms;
        FirmQuotes {
            offers,
            masks: offers + 2 * firms,
        }
    }

    pub(in crate::simulation::economy) fn offer(&self, member: u32) -> u32 {
        self.offers + 2 * member
    }

    fn mask(&self, rotation: u32, word: u32) -> u32 {
        self.masks + rotation * QUOTE_MASK_WORDS + word
    }
}

#[cube]
pub(in crate::simulation::economy) fn index_offer_prices(
    economy: &EconomyView<'_>,
    spatial: &mut [u32],
    parameters: &EconomyParameters,
) {
    let config = economy.ledger.config;
    let cells = config.spatial_side * config.spatial_side;
    let segment = 2 * (cells + 1 + config.actors);
    let firms = spatial[(segment + cells) as usize];
    let first = CUBE_POS as u32 * DISCOVERY_BLOCK;
    let mut member = first + UNIT_POS;
    while member < min(first + DISCOVERY_BLOCK, firms) {
        let seller = spatial[(segment + cells + 1 + member) as usize];
        let quote = FirmQuotes::new(cells, firms).offer(member);
        match comptime!(parameters.market) {
            MarketFlag::Consumer(ConsumerMarket::Fuel) => {
                spatial[quote as usize] =
                    offer_price(economy, seller, comptime!(Good::Coal as u32));
                spatial[(quote + 1) as usize] =
                    offer_price(economy, seller, comptime!(Good::Wood as u32));
            }
            _ => spatial[quote as usize] = offer_price(economy, seller, parameters.traded_good()),
        }
        member += CUBE_DIM;
    }
}

#[cube]
pub(in crate::simulation::economy) fn index_quote_masks(
    spatial: &mut [u32],
    firms: u32,
    #[comptime] market: MarketFlag,
) {
    let rotation = ABSOLUTE_POS as u32;
    let quotes = FirmQuotes::new(1, firms);
    if rotation < firms {
        let samples = min(firms, HOUSEHOLD_SEARCH_SAMPLES);
        for word in 0..QUOTE_MASK_WORDS {
            let mut mask = 0u32;
            for sample in word * 32..min(word * 32 + 32, samples) {
                let offset = quote_sample_offset(firms, sample);
                let shifted = rotation + offset;
                let member = select(shifted < firms, shifted, shifted - firms);
                let quote = quotes.offer(member);
                let mut available = spatial[quote as usize];
                if comptime!(matches!(market, MarketFlag::Consumer(ConsumerMarket::Fuel))) {
                    available |= spatial[(quote + 1) as usize];
                }
                mask |= u32::cast_from(available > 0) << (sample % 32);
            }
            spatial[quotes.mask(rotation, word) as usize] = mask;
        }
    }
}

#[cube]
fn quote_sample_offset(length: u32, sample: u32) -> u32 {
    let width = comptime!(HOUSEHOLD_SEARCH_SAMPLES);
    let spaced = sample * (length / width) + sample * (length % width) / width;
    select(length < width, sample, spaced)
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct QuoteSamples {
    length: u32,
    origin: u32,
    rotation: u32,
    offset_fits: bool,
}

#[cube]
impl QuoteSamples {
    pub fn new(length: u32, origin: u32) -> Self {
        let divisor = max(length, 1);
        QuoteSamples {
            length,
            origin,
            rotation: origin % divisor,
            offset_fits: origin <= u32::MAX - (divisor - 1),
        }
    }

    pub fn count(&self) -> u32 {
        min(self.length, HOUSEHOLD_SEARCH_SAMPLES)
    }

    pub fn member(&self, sample: u32) -> u32 {
        let offset = quote_sample_offset(self.length, sample);
        if self.offset_fits {
            let shifted = self.rotation + offset;
            select(shifted < self.length, shifted, shifted - self.length)
        } else {
            (offset + self.origin) % self.length
        }
    }

    pub fn next(&self, spatial: &[u32], start: u32, #[comptime] source: QuoteSource) -> u32 {
        let mut next = start;
        if comptime!(source == QuoteSource::Masked) && self.offset_fits {
            let quotes = FirmQuotes::new(1, self.length);
            let samples = self.count();
            let mut word = start / 32;
            let mut bits = 0u32;
            while bits == 0 && word * 32 < samples {
                bits = spatial[quotes.mask(self.rotation, word) as usize];
                if word == start / 32 {
                    bits &= u32::MAX << (start % 32);
                }
                word += 1;
            }
            next = select(bits == 0, samples, (word - 1) * 32 + bits.trailing_zeros());
        }
        next
    }
}

#[derive(CubeType)]
pub(in crate::simulation::economy) struct ReplacementMarket<'a> {
    spatial: &'a [u32],
    control: &'a [Atomic<u32>],
    day: u32,
}

#[cube]
impl<'a> ReplacementMarket<'a> {
    pub(in crate::simulation::economy) fn new(
        spatial: &'a [u32],
        control: &'a [Atomic<u32>],
        day: u32,
    ) -> Self {
        ReplacementMarket::<'a> {
            spatial,
            control,
            day,
        }
    }

    pub(in crate::simulation::economy) fn refresh(
        &self,
        economy: &mut EconomyView<'_>,
        buyer: u32,
    ) {
        let control = self.control;
        let config = economy.ledger.config;
        if buyer < config.actors {
            let active = !economy
                .actors
                .kind(buyer)
                .is(comptime!(ActorKind::Household))
                && economy.ledger.number(Field::EnterpriseStatus, buyer)
                    < comptime!(EnterpriseState::Liquidating as u32);
            if active {
                for input in 0..3u32 {
                    let field = match input {
                        0 | 1 => comptime!(Field::FirstInputCost as u32) + input,
                        _ => u32::new(comptime!(Field::ReplacementCost as i64)),
                    };
                    let good = match input {
                        0 | 1 => economy
                            .ledger
                            .number_at(comptime!(Field::FirstInput as u32) + input, buyer),
                        _ => {
                            let distributor = economy.ledger.number(Field::EnterpriseModel, buyer)
                                == comptime!(EnterpriseModel::Distributor as u32);
                            select(distributor, economy.actors.kind(buyer).output(), 0u32)
                        }
                    };
                    if good > 0 && !(input == 2 && good == comptime!(Good::Grain as u32)) {
                        let stock = economy
                            .ledger
                            .stock_with_transit(good, buyer)
                            .record(control);
                        let batch = max(economy.ledger.number(Field::InputTicks, buyer), 1);
                        let stocked =
                            comptime!(config.economy == crate::domain::Economy::Industrial)
                                && input < 2
                                && !lt(stock, ticks(batch, 0))
                                && economy.ledger.number_at(field, buyer) > 0;
                        let orders = !equal(economy.ledger.value(Field::Sales, buyer), zero())
                            || !equal(economy.ledger.value(Field::Unfilled, buyer), zero());
                        if !stocked || orders {
                            let current = self.quote(
                                &*economy,
                                buyer,
                                good,
                                comptime!(QuoteSource::FromLedger),
                            );
                            if current > 0 {
                                economy.ledger.set_number_at(field, buyer, current);
                            }
                        }
                    }
                }
            }
        }
    }

    pub(in crate::simulation::economy) fn quote(
        &self,
        economy: &EconomyView<'_>,
        buyer: u32,
        good: u32,
        #[comptime] quotes: QuoteSource,
    ) -> u32 {
        let spatial = self.spatial;
        let control = self.control;
        let config = economy.ledger.config;
        let side = config.spatial_side;
        let cells = side * side;
        let segment = 2 * (cells + 1 + config.actors);
        let mut buyer_access = 0u32;
        if config.railway.has_routes() {
            buyer_access = config.railway.access[buyer as usize * 2 + 1];
        }
        let columns = search::SearchAxis::around(economy.actors.x(buyer), config);
        let rows = search::SearchAxis::around(economy.actors.y(buyer), config);
        let mut quote = 0u32;
        let mut best = ticks(u32::MAX, u32::MAX);
        let mut row = rows.first;
        while row <= rows.last {
            let mut column = columns.first;
            while column <= columns.last {
                let cell = row * side + column;
                let begin = spatial[(segment + cell) as usize];
                let length = spatial[(segment + cell + 1) as usize] - begin;
                let samples = QuoteSamples::new(length, self.day + buyer);
                let mut sample = samples.next(spatial, 0, comptime!(quotes));
                while sample < samples.count() {
                    let rotated = samples.member(sample);
                    let seller = spatial[(segment + cells + 1 + begin + rotated) as usize];
                    let price = match comptime!(quotes) {
                        QuoteSource::Indexed | QuoteSource::Masked => {
                            let firms = spatial[(segment + cells) as usize];
                            spatial[FirmQuotes::new(cells, firms).offer(begin + rotated) as usize]
                        }
                        QuoteSource::FromLedger => offer_price(economy, seller, good),
                    };
                    if price > 0
                        && seller != buyer
                        && household_production::counterparty(economy, buyer, seller, good)
                    {
                        let trip = economy.actors.distance(buyer, seller);
                        let construction = good == comptime!(Good::Construction as u32);
                        if trip <= config.search_radius {
                            let days = if construction {
                                0u32
                            } else {
                                railway::DeliveryRequest {
                                    buyer,
                                    seller,
                                    trip,
                                    buyer_access,
                                }
                                .days(config)
                            };
                            let delivered = select(
                                construction,
                                price,
                                cartage::delivered_price(config, price, trip, days),
                            );
                            let score =
                                plus(ticks(delivered, 0), times32(days, config.delivery_penalty))
                                    .record(control);
                            if lt(score, best) {
                                quote = delivered;
                                best = score;
                            }
                        }
                    }
                    sample = samples.next(spatial, sample + 1, comptime!(quotes));
                }
                column += 1;
            }
            row += 1;
        }
        quote
    }
}

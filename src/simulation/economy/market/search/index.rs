use super::*;

const FIELDS: usize = 10;
pub(super) const BLOCK: u32 = 32;
pub(super) const HOUSEHOLD_STRIDE_FIELD: u32 = 6;
pub(super) const HOUSEHOLD_X_FIELD: u32 = 8;
pub(super) const HOUSEHOLD_Y_FIELD: u32 = 9;

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub(super) enum SupplySource {
    State,
    Grain,
    Index,
}

#[derive(Clone, Copy)]
enum BlockField {
    StockedPrice,
    EmptyPrice,
    MinX,
    MaxX,
    MinY,
    MaxY,
    Stocked,
    MinAccess,
    MaximumAvailable,
    MaximumQuote,
}

#[derive(Clone, Copy)]
enum SupplyField {
    Quote,
    AvailableLow,
    AvailableHigh,
    Flags,
}

#[derive(CubeType, Clone, Copy)]
pub(super) struct Supply {
    pub quote: u32,
    pub available: Ticks,
    pub offered: bool,
    pub producer: bool,
}

#[cube]
impl Supply {
    pub fn for_search(
        economy: &EconomyView<'_>,
        scratch: &[u32],
        parameters: &EconomyParameters,
        member: u32,
        actor: u32,
        #[comptime] source: SupplySource,
    ) -> Self {
        if comptime!(
            matches!(source, SupplySource::Index | SupplySource::Grain)
                || matches!(
                    parameters.market,
                    MarketFlag::Consumer(_) | MarketFlag::Household
                )
        ) {
            Self::load(scratch, member)
        } else {
            Self::new(economy, actor, parameters.traded_good())
        }
    }

    pub fn new(economy: &EconomyView<'_>, actor: u32, good: u32) -> Self {
        let kind = economy.actors.kind(actor);
        let producer = !kind.is(comptime!(ActorKind::Household))
            && (kind.output() == good || pricing::distributor(&economy.ledger, actor));
        let supplier = household_production::supplier(economy, actor, good);
        let terms = household_production::TradeFields::new(
            &economy.actors,
            economy.ledger.config,
            actor,
            good,
        );
        let available = positive(
            economy.ledger.value_at(terms.stock, actor),
            household_production::reserve(economy, actor, good),
        );
        Supply {
            quote: economy.ledger.number_at(terms.price, actor),
            available,
            offered: supplier && !equal(available, zero()),
            producer,
        }
    }

    pub(super) fn store(&self, scratch: &mut [u32], member: u32) {
        let base = member * 4;
        scratch[(base + comptime!(SupplyField::Quote as u32)) as usize] = self.quote;
        scratch[(base + comptime!(SupplyField::AvailableLow as u32)) as usize] =
            self.available.extract(LOW_WORD);
        scratch[(base + comptime!(SupplyField::AvailableHigh as u32)) as usize] =
            self.available.extract(HIGH_WORD);
        scratch[(base + comptime!(SupplyField::Flags as u32)) as usize] =
            u32::cast_from(self.offered) | (u32::cast_from(self.producer) << 1);
    }

    pub fn load(scratch: &[u32], member: u32) -> Self {
        let base = member * 4;
        let flags = scratch[(base + comptime!(SupplyField::Flags as u32)) as usize];
        Supply {
            quote: scratch[(base + comptime!(SupplyField::Quote as u32)) as usize],
            available: ticks(
                scratch[(base + comptime!(SupplyField::AvailableLow as u32)) as usize],
                scratch[(base + comptime!(SupplyField::AvailableHigh as u32)) as usize],
            ),
            offered: flags & 1 != 0,
            producer: flags & 2 != 0,
        }
    }
}

#[cube]
fn reduce_minimum(value: u32, #[comptime] reduction: parallel::ScanMode) -> u32 {
    match comptime!(reduction) {
        parallel::ScanMode::Shared => parallel::reduce::<u32>(value, comptime!(Reduction::Minimum)),
        parallel::ScanMode::Plane => plane_min(value),
    }
}

#[derive(CubeType)]
struct SupplierBlockStats {
    stocked_price: u32,
    empty_price: u32,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
    stocked: u32,
    min_access: u32,
    maximum_available: u32,
    maximum_quote: u32,
}

#[cube]
impl SupplierBlockStats {
    fn new() -> Self {
        let missing = u32::new(comptime!(i64::from(u32::MAX)));
        SupplierBlockStats {
            stocked_price: missing,
            empty_price: missing,
            min_x: missing,
            max_x: 0,
            min_y: missing,
            max_y: 0,
            stocked: 0,
            min_access: missing,
            maximum_available: 0,
            maximum_quote: 0,
        }
    }

    fn collect(&mut self, supply: &Supply, x: u32, y: u32, access: u32) {
        let quote = supply.quote;
        let offered = supply.offered;
        let price = select(quote > 0, quote, u32::MAX);
        self.stocked_price = min(self.stocked_price, select(offered, price, u32::MAX));
        self.empty_price = min(
            self.empty_price,
            select(supply.producer && !offered, price, u32::MAX),
        );
        self.min_x = min(self.min_x, x);
        self.max_x = max(self.max_x, x);
        self.min_y = min(self.min_y, y);
        self.max_y = max(self.max_y, y);
        self.stocked = max(self.stocked, u32::cast_from(offered));
        self.min_access = min(self.min_access, access);
        self.maximum_available = max(
            self.maximum_available,
            select(
                supply.available.extract(HIGH_WORD) > 0,
                u32::MAX,
                supply.available.extract(LOW_WORD),
            ),
        );
        self.maximum_quote = max(self.maximum_quote, select(offered, quote, 0));
    }

    fn reduce(&mut self, config: &EconomyConfig, #[comptime] reduction: parallel::ScanMode) {
        self.stocked_price = reduce_minimum(self.stocked_price, comptime!(reduction));
        self.empty_price = reduce_minimum(self.empty_price, comptime!(reduction));
        self.min_x = reduce_minimum(self.min_x, comptime!(reduction));
        self.max_x = !reduce_minimum(!self.max_x, comptime!(reduction));
        self.min_y = reduce_minimum(self.min_y, comptime!(reduction));
        self.max_y = !reduce_minimum(!self.max_y, comptime!(reduction));
        self.stocked = !reduce_minimum(!self.stocked, comptime!(reduction));
        self.maximum_available = !reduce_minimum(!self.maximum_available, comptime!(reduction));
        self.maximum_quote = !reduce_minimum(!self.maximum_quote, comptime!(reduction));
        if config.railway.has_routes() {
            self.min_access = reduce_minimum(self.min_access, comptime!(reduction));
        }
    }

    fn store(&self, scratch: &mut [u32], actors: u32, block: u32) {
        let base = 6 * actors as usize + block as usize * FIELDS;
        scratch[base + comptime!(BlockField::StockedPrice as usize)] = self.stocked_price;
        scratch[base + comptime!(BlockField::EmptyPrice as usize)] = self.empty_price;
        scratch[base + comptime!(BlockField::MinX as usize)] = self.min_x;
        scratch[base + comptime!(BlockField::MaxX as usize)] = self.max_x;
        scratch[base + comptime!(BlockField::MinY as usize)] = self.min_y;
        scratch[base + comptime!(BlockField::MaxY as usize)] = self.max_y;
        scratch[base + comptime!(BlockField::Stocked as usize)] = self.stocked;
        scratch[base + comptime!(BlockField::MinAccess as usize)] = self.min_access;
        scratch[base + comptime!(BlockField::MaximumAvailable as usize)] = self.maximum_available;
        scratch[base + comptime!(BlockField::MaximumQuote as usize)] = self.maximum_quote;
    }
}

#[cube]
impl SearchInput {
    pub(super) fn build_supplier_index(&mut self) {
        let view = EconomyView::new(&mut self.state, &self.actors, &self.config);
        let economy = &view;
        let spatial: &mut [u32] = &mut self.spatial;
        let scratch: &mut [u32] = &mut self.scratch;
        let searches: &mut [u32] = &mut self.searches;
        let successes: &[Atomic<u32>] = &self.successes;
        let control: &[Atomic<u32>] = &self.control;
        let parameters = &self.parameters;
        let collection = comptime!(self.program.collection);
        let config = economy.ledger.config;
        let reduction = comptime!(parameters.scan);
        let cells = config.spatial_side * config.spatial_side;
        let count = spatial[cells as usize];
        let start = CUBE_POS as u32 * (DISCOVERY_BLOCK / BLOCK);
        let end = min(start + DISCOVERY_BLOCK / BLOCK, ceildiv(count, BLOCK));
        let width = match comptime!(reduction) {
            parallel::ScanMode::Plane => min(PLANE_DIM, CUBE_DIM),
            parallel::ScanMode::Shared => CUBE_DIM,
        };
        let lane = match comptime!(reduction) {
            parallel::ScanMode::Plane => UNIT_POS_PLANE,
            parallel::ScanMode::Shared => UNIT_POS,
        };
        let step = ceildiv(CUBE_DIM, width);
        let mut block = start
            + match comptime!(reduction) {
                parallel::ScanMode::Plane => PLANE_POS,
                parallel::ScanMode::Shared => 0u32,
            };
        let good = parameters.traded_good();
        let mut minimum_quote = u32::new(comptime!(i64::from(u32::MAX)));
        let mut minimum_eligible = u32::new(comptime!(i64::from(u32::MAX)));
        while block < end {
            let mut stats = SupplierBlockStats::new();
            let mut member = block * BLOCK + lane;
            while member < min((block + 1) * BLOCK, count) {
                let actor = spatial[(cells + 1 + member) as usize];
                if comptime!(
                    parameters.empty_orders
                        == crate::simulation::economy::parameters::EmptyOrderMode::FeedbackOnly
                        && matches!(parameters.market, MarketFlag::Consumer(_))
                ) {
                    successes[orders::filtered_count(config.actors, actor) as usize].store(0);
                    if comptime!(collection == OrderCollection::ConsumerMask) {
                        let slot = filter::prefix_slot(successes, config.actors, actor);
                        successes[slot].store(0);
                        successes[slot + 1].store(0);
                    }
                }
                let supply = Supply::new(economy, actor, good);
                supply.store(scratch, member);
                let quote = supply.quote;
                if good == comptime!(Good::Bread as u32) {
                    minimum_quote = min(minimum_quote, quote);
                }
                let offered = supply.offered;
                if comptime!(matches!(parameters.market, MarketFlag::Input(_)))
                    && good == comptime!(Good::Bread as u32)
                    && quote > 0
                    && (offered || supply.producer)
                {
                    minimum_eligible = min(minimum_eligible, quote);
                }
                let x = economy.actors.x(actor);
                let y = economy.actors.y(actor);
                if comptime!(parameters.market == MarketFlag::Household) {
                    searches[(HOUSEHOLD_X_FIELD * config.actors + member) as usize] = x;
                    searches[(HOUSEHOLD_Y_FIELD * config.actors + member) as usize] = y;
                    let cell =
                        y / config.search_radius * config.spatial_side + x / config.search_radius;
                    let begin = spatial[cell as usize];
                    let length = spatial[(cell + 1) as usize] - begin;
                    let offset = member - begin;
                    if length > HOUSEHOLD_SEARCH_SAMPLES && offset < length / 2 {
                        searches[(HOUSEHOLD_STRIDE_FIELD * config.actors + member) as usize] =
                            coprime_stride(length, offset);
                    }
                }
                if comptime!(matches!(parameters.market, MarketFlag::Consumer(_))) {
                    scratch[(5 * config.actors + member) as usize] = empty::key(config, x, y);
                }
                let mut access = u32::new(comptime!(i64::from(u32::MAX)));
                if config.railway.has_routes() {
                    access = config.railway.access[actor as usize * 2 + 1];
                }
                stats.collect(&supply, x, y, access);
                member += width;
            }
            stats.reduce(config, comptime!(reduction));
            if lane == 0 {
                stats.store(scratch, config.actors, block);
            }
            block += step;
        }
        if comptime!(matches!(parameters.market, MarketFlag::Input(_)))
            && good == comptime!(Good::Bread as u32)
        {
            let minimum = parallel::reduce::<u32>(minimum_eligible, comptime!(Reduction::Minimum));
            if UNIT_POS == 0 && minimum < control[INPUT_MINIMUM_QUOTE_WORD].load() {
                control[INPUT_MINIMUM_QUOTE_WORD].fetch_min(minimum);
            }
        }
        if good == comptime!(Good::Bread as u32) {
            let minimum = parallel::reduce::<u32>(minimum_quote, comptime!(Reduction::Minimum));
            if UNIT_POS == 0 && CUBE_POS as u32 * DISCOVERY_BLOCK < count {
                let base = 3 * (cells + 1 + config.actors);
                spatial[(base + CUBE_POS as u32) as usize] = minimum;
            }
        }
    }
}

#[derive(CubeType)]
pub(super) struct SupplierBlock<'a> {
    config: &'a EconomyConfig,
    words: &'a [u32],
    base: usize,
}

#[cube]
impl<'a> SupplierBlock<'a> {
    pub(super) fn new(words: &'a [u32], config: &'a EconomyConfig, block: u32) -> Self {
        SupplierBlock::<'a> {
            config,
            words,
            base: 6 * config.actors as usize + block as usize * FIELDS,
        }
    }

    fn field(&self, #[comptime] field: BlockField) -> u32 {
        self.words[self.base + comptime!(field as usize)]
    }

    fn distance(&self, buyer: &BuyerLocation) -> u32 {
        buyer.distance_to_bounds(
            self.field(comptime!(BlockField::MinX)),
            self.field(comptime!(BlockField::MaxX)),
            self.field(comptime!(BlockField::MinY)),
            self.field(comptime!(BlockField::MaxY)),
        )
    }

    fn delivery_days(&self, buyer: &BuyerLocation, distance: u32) -> u32 {
        let config = self.config;
        let mut days = max(ceildiv(distance, config.shipment_speed), 1);
        if config.railway.has_routes() {
            let seller_access = self.field(comptime!(BlockField::MinAccess));
            days = min(
                days,
                ceildiv(buyer.access + seller_access, config.shipment_speed) + 1,
            );
        }
        days
    }

    pub(super) fn maximum_available(&self) -> u32 {
        self.field(comptime!(BlockField::MaximumAvailable))
    }

    pub(super) fn partial_input_price_floor(
        &self,
        buyer: &BuyerLocation,
        maximum_delivery_cost: Ticks,
    ) -> u32 {
        let minimum_quote = self.field(comptime!(BlockField::StockedPrice));
        let maximum_quote = self.field(comptime!(BlockField::MaximumQuote));
        let days = self.delivery_days(buyer, self.distance(buyer));
        let lower = plus(
            ticks(minimum_quote, 0),
            times32(days, self.config.delivery_penalty),
        )
        .value;
        let upper = plus(ticks(maximum_quote, 0), maximum_delivery_cost).value;
        select(upper.extract(HIGH_WORD) == 0, lower.extract(LOW_WORD), 0)
    }

    pub(super) fn skip(
        &self,
        buyer: &BuyerLocation,
        cash: Ticks,
        chosen: u32,
        empty_score: Ticks,
        visibility: u32,
    ) -> bool {
        let stocked_price = self.field(comptime!(BlockField::StockedPrice));
        let empty_price = self.field(comptime!(BlockField::EmptyPrice));
        let stocked = self.field(comptime!(BlockField::Stocked));
        let distance = self.distance(buyer);
        let days = self.delivery_days(buyer, distance);
        let delivery = times32(days, self.config.delivery_penalty);
        let lower_score = plus(ticks(empty_price, 0), delivery).value;
        let stock_possible = stocked_price != u32::MAX && !lt(cash, ticks(stocked_price, 0));
        let empty_improves = chosen == NO_ACTOR
            && empty_price != u32::MAX
            && !lt(cash, ticks(empty_price, 0))
            && lt(lower_score, empty_score);
        let unseen_stock = stocked > 0 && visibility == 0;
        distance > self.config.search_radius || !(stock_possible || empty_improves || unseen_stock)
    }

    pub(super) fn skip_stocked(&self, buyer: &BuyerLocation, cash: Ticks) -> bool {
        let price = self.field(comptime!(BlockField::StockedPrice));
        self.distance(buyer) > self.config.search_radius
            || price == u32::MAX
            || lt(cash, ticks(price, 0))
    }
}

use crate::simulation::economy::*;

#[derive(CubeType, Clone, Copy)]
struct KernelLeg {
    field: u32,
    actor: u32,
    debit: Ticks,
    credit: Ticks,
    authorization: KernelAuthorization,
    guard: Ticks,
}

#[derive(CubeType)]
struct EventBuilder<'a> {
    bundles: &'a mut [u32],
    config: &'a EconomyConfig,
    position: usize,
}
#[cube]
impl<'a> EventBuilder<'a> {
    fn new(bundles: &'a mut [u32], config: &'a EconomyConfig, event: u32) -> Self {
        let position = event as usize * EVENT_WORDS;
        for slot in 0u32..MAX_LEGS as u32 {
            for word in 0..LEG_WORDS {
                bundles[position + slot as usize * LEG_WORDS + word] =
                    select(word == LegLayout::COORDINATE, EMPTY, 0u32);
            }
        }
        EventBuilder::<'a> {
            bundles,
            config,
            position,
        }
    }
    fn leg(&mut self, leg: KernelLeg) {
        let base = self.position;
        self.bundles[base + LegLayout::COORDINATE] = self.config.coordinate(leg.field, leg.actor);
        self.bundles[base + LegLayout::DEBIT_LOW] = leg.debit.extract(LOW_WORD);
        self.bundles[base + LegLayout::DEBIT_HIGH] = leg.debit.extract(HIGH_WORD);
        self.bundles[base + LegLayout::CREDIT_LOW] = leg.credit.extract(LOW_WORD);
        self.bundles[base + LegLayout::CREDIT_HIGH] = leg.credit.extract(HIGH_WORD);
        self.bundles[base + LegLayout::AUTHORIZATION] = leg.authorization.bits();
        self.bundles[base + LegLayout::GUARD_LOW] = leg.guard.extract(LOW_WORD);
        self.bundles[base + LegLayout::GUARD_HIGH] = leg.guard.extract(HIGH_WORD);
        self.position += LEG_WORDS;
    }

    fn transfer(&mut self, field: u32, actor: u32, debit: Ticks, credit: Ticks) {
        self.leg(KernelLeg {
            field,
            actor,
            debit,
            credit,
            authorization: KernelAuthorization::new(comptime!(AuthorizationFlags::empty())),
            guard: ticks(0u32, 0u32),
        });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::simulation::economy) enum EventStorage {
    Buffer,
    Local,
}

#[cube]
pub(in crate::simulation::economy) fn construct(
    economy: &mut EconomyView<'_>,
    proposals: &mut [u32],
    bundles: &mut [u32],
    control: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
    #[comptime] storage: EventStorage,
) {
    let config = economy.ledger.config;
    if id < parameters.count {
        let actor = id + parameters.offset;
        let other = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let mut route = railway::FreightRoute::by_road(&economy.actors, config, actor, actor);
        if parameters.traded_good() > 0 && other < config.actors {
            let planned = railway::FreightRoute::by_road(&economy.actors, config, actor, other);
            route.replace(&planned);
        }
        let event_id = match comptime!(storage) {
            EventStorage::Buffer => id,
            EventStorage::Local => 0,
        };
        let mut event = EventBuilder::new(bundles, config, event_id);
        let quantity = ticks(
            proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW],
            proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH],
        );
        let trip = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::DISTANCE];
        let zero = ticks(0u32, 0u32);
        if other != NO_ACTOR {
            if comptime!(storage == EventStorage::Buffer) {
                control[CONSTRUCTED_EVENTS_WORD].fetch_add(1u32);
            }
            match comptime!(parameters.market) {
                MarketFlag::Labor => {
                    let shifts = employment::shifts(config, quantity);
                    let labor = economy.ledger.number(Field::LaborMinutes, other);
                    let mut offered_wage = economy.ledger.number(Field::Wage, other);
                    if comptime!(config.bargaining.enabled == Feature::Enabled) {
                        offered_wage = quantity.extract(LOW_WORD);
                    }
                    let wage = times32(offered_wage, labor * shifts);
                    event.transfer(comptime!(Field::Cash as u32), other, wage, zero);
                    event.transfer(comptime!(Field::Escrow as u32), other, zero, wage);
                    event.transfer(
                        comptime!(Field::Slots as u32),
                        other,
                        ticks(shifts, 0u32),
                        zero,
                    );
                    event.transfer(
                        comptime!(Field::Time as u32),
                        actor,
                        ticks(
                            (labor + ceildiv(trip, config.commute_speed) * COMMUTE_LEGS) * shifts,
                            0u32,
                        ),
                        zero,
                    );
                    event.transfer(
                        comptime!(Field::Employer as u32),
                        actor,
                        zero,
                        ticks(other + 1u32, 0u32),
                    );
                    event.transfer(comptime!(Field::Due as u32), actor, zero, wage);
                    event.leg(KernelLeg {
                        field: comptime!(Field::Employed as u32),
                        actor,
                        debit: zero,
                        credit: ticks(shifts, 0u32),
                        authorization: KernelAuthorization::new(comptime!(
                            AuthorizationFlags::EQUAL
                        )),
                        guard: zero,
                    });
                    event.transfer(
                        comptime!(Field::Hours as u32),
                        actor,
                        zero,
                        ticks(labor * shifts, 0u32),
                    );
                    event.transfer(
                        comptime!(Field::Pipeline as u32),
                        other,
                        zero,
                        ticks(
                            economy.ledger.number(Field::OutputTicks, other) * shifts,
                            0u32,
                        ),
                    );
                    event.leg(KernelLeg {
                        field: comptime!(Field::Wage as u32),
                        actor: other,
                        debit: zero,
                        credit: zero,
                        authorization: KernelAuthorization::new(comptime!(
                            AuthorizationFlags::EQUAL
                        )),
                        guard: economy.ledger.value(Field::Wage, other),
                    });
                    let mut total_used = zero;
                    for input in 0u32..2 {
                        let good = economy
                            .ledger
                            .number_at(comptime!(Field::FirstInput as u32) + input, other);
                        if good > 0 {
                            let amount =
                                times(economy.ledger.value(Field::InputTicks, other), shifts)
                                    .record(control);
                            event.transfer(good, other, amount, zero);
                            event.transfer(USED_FIELDS as u32 + good - 1, other, zero, amount);
                            total_used = plus(total_used, amount).record(control);
                        }
                    }
                    event.transfer(comptime!(Field::Used as u32), other, zero, total_used);
                }
                MarketFlag::Payroll => {
                    let mut net = quantity;
                    let prior = economy.ledger.value(Field::GrossIncome, actor);
                    event.leg(KernelLeg {
                        field: comptime!(Field::GrossIncome as u32),
                        actor,
                        debit: zero,
                        credit: quantity,
                        authorization: KernelAuthorization::new(comptime!(
                            AuthorizationFlags::EQUAL
                        )),
                        guard: prior,
                    });
                    if comptime!(config.payroll == Feature::Enabled) {
                        let accrued = plus(prior, quantity).record(control);
                        let members = employment::payroll_members(&economy.actors, config, actor);
                        let before = PayrollAccrual::new(prior, members, config, control);
                        let after = PayrollAccrual::new(accrued, members, config, control);
                        net = minus(after.net, before.net);
                        event.leg(KernelLeg {
                            field: comptime!(Field::PayrollReserve as u32),
                            actor,
                            debit: before.reserve,
                            credit: after.reserve,
                            authorization: KernelAuthorization::new(comptime!(
                                AuthorizationFlags::EQUAL
                            )),
                            guard: before.reserve,
                        });
                        event.transfer(
                            comptime!(Field::TaxTreasury as u32),
                            actor,
                            zero,
                            minus(after.tax, before.tax),
                        );
                        event.transfer(
                            comptime!(Field::ChargeTreasury as u32),
                            actor,
                            zero,
                            minus(after.charge, before.charge),
                        );
                    }
                    event.transfer(comptime!(Field::Escrow as u32), other, quantity, zero);
                    event.transfer(comptime!(Field::Due as u32), actor, quantity, zero);
                    event.transfer(comptime!(Field::Cash as u32), actor, zero, net);
                    event.leg(KernelLeg {
                        field: comptime!(Field::Employer as u32),
                        actor,
                        debit: zero,
                        credit: zero,
                        authorization: KernelAuthorization::new(comptime!(
                            AuthorizationFlags::EQUAL
                        )),
                        guard: ticks(other + 1u32, 0u32),
                    });
                }
                MarketFlag::Distribution => {
                    event.transfer(comptime!(Field::Cash as u32), actor, quantity, zero);
                    event.transfer(comptime!(Field::Cash as u32), other, zero, quantity);
                }
                MarketFlag::Input(InputMarket::Construction) => {
                    let price = economy.ledger.number(Field::Price, other);
                    let cost = times(quantity, price).record(control);
                    event.transfer(comptime!(Field::Cash as u32), actor, cost, zero);
                    event.transfer(comptime!(Field::Cash as u32), other, zero, cost);
                    event.transfer(comptime!(Field::Pipeline as u32), other, quantity, zero);
                    event.transfer(comptime!(Field::Construction as u32), actor, zero, quantity);
                    event.transfer(comptime!(Field::Sales as u32), other, zero, quantity);
                    event.transfer(comptime!(Field::Purchased as u32), actor, zero, quantity);
                    if economy.actors.kind(actor).output() == comptime!(Good::Housing as u32) {
                        event.transfer(
                            comptime!(Field::FirstInputCost as u32),
                            actor,
                            economy.ledger.value(Field::FirstInputCost, actor),
                            ticks(price, 0),
                        );
                    }
                }
                _ => {
                    let good = parameters.traded_good();
                    let terms = household_production::TradeFields::new(
                        &economy.actors,
                        economy.ledger.config,
                        other,
                        good,
                    );
                    let cost = times(quantity, economy.ledger.number_at(terms.price, other))
                        .record(control);
                    let freight = times(quantity, route.fee).record(control);
                    let trip = economy.actors.distance(actor, other);
                    let slot =
                        cartage::booking_field(config, good, parameters.day, route.days, trip);
                    if trip > 0 {
                        event.transfer(
                            comptime!(Field::CartagePayable as u32),
                            actor,
                            zero,
                            times(quantity, cartage::rate(config, route.days)).record(control),
                        );
                        event.transfer(comptime!(Field::RoadFreight as u32), actor, zero, quantity);
                    }
                    event.transfer(comptime!(Field::Cash as u32), actor, cost, zero);
                    event.transfer(comptime!(Field::Cash as u32), other, freight, cost);
                    let retained = household_production::reserve(&*economy, other, good);
                    event.leg(KernelLeg {
                        field: good,
                        actor: other,
                        debit: quantity,
                        credit: zero,
                        authorization: KernelAuthorization::new(comptime!(
                            AuthorizationFlags::AT_LEAST
                        )),
                        guard: plus(retained, quantity).record(control),
                    });
                    config.mark_ring_occupancy(actor, good - 1);
                    event.transfer(slot, actor, zero, quantity);
                    let delivered = cartage::road_price(
                        config,
                        control,
                        economy.ledger.value_at(terms.price, other),
                        trip,
                        route.days,
                    );
                    if good == comptime!(Good::Bread as u32)
                        && economy
                            .actors
                            .kind(actor)
                            .is(comptime!(ActorKind::Household))
                    {
                        event.transfer(
                            comptime!(Field::FoodPrice as u32),
                            actor,
                            economy.ledger.value(Field::FoodPrice, actor),
                            delivered,
                        );
                    }
                    for input in 0u32..2 {
                        if economy
                            .ledger
                            .number_at(comptime!(Field::FirstInput as u32) + input, actor)
                            == good
                        {
                            let cost_field = comptime!(Field::FirstInputCost as u32) + input;
                            event.transfer(
                                cost_field,
                                actor,
                                economy.ledger.value_at(cost_field, actor),
                                delivered,
                            );
                        }
                    }
                    if pricing::distributor(&economy.ledger, actor) {
                        let cost_field =
                            pricing::cost_field(config, economy.actors.kind(actor).output(), good);
                        event.transfer(
                            cost_field,
                            actor,
                            economy.ledger.value_at(cost_field, actor),
                            delivered,
                        );
                    }
                    event.transfer(TRANSIT_FIELDS as u32 + good - 1, actor, zero, quantity);
                    event.transfer(comptime!(Field::Transit as u32), actor, zero, quantity);
                    event.transfer(terms.sales, other, zero, quantity);
                    event.transfer(comptime!(Field::Purchased as u32), actor, zero, quantity);
                    event.leg(KernelLeg {
                        field: terms.price,
                        actor: other,
                        debit: zero,
                        credit: zero,
                        authorization: KernelAuthorization::new(comptime!(
                            AuthorizationFlags::EQUAL
                        )),
                        guard: economy.ledger.value_at(terms.price, other),
                    });
                }
            }
        }
    }
}

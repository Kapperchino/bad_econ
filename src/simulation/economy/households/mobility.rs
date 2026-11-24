use crate::simulation::economy::*;

#[repr(u32)]
enum WorkReview {
    Inactive,
    Employer,
    Vacancies,
}

#[derive(CubeType)]
struct LocalNeeds {
    bread: u32,
    coal: u32,
    wood: u32,
}

#[cube]
impl LocalNeeds {
    fn cached(quotes: &[u32], config: &EconomyConfig, destination: u32) -> Self {
        let base = 4 * config.actors + 3 * destination;
        LocalNeeds {
            bread: quotes[base as usize],
            coal: quotes[base as usize + 1],
            wood: quotes[base as usize + 2],
        }
    }

    fn new(economy: &mut EconomyView<'_>, spatial: &[u32], destination: u32, day: u32) -> Self {
        let config = economy.ledger.config;
        let cells = config.spatial_side * config.spatial_side;
        let segment = 2 * (cells + 1 + config.actors);
        let length = spatial[(segment + cells) as usize];
        let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
        let mut needs = LocalNeeds {
            bread: config.maximum_price + 1,
            coal: 2 * config.maximum_price + 1,
            wood: config.maximum_price + 1,
        };
        for sample in 0..samples {
            let offset = sample * (length / samples) + sample * (length % samples) / samples;
            let seller =
                spatial[(segment + cells + 1 + (offset + day + destination) % length) as usize];
            let output = economy.actors.kind(seller).output();
            let dealer = pricing::distributor(&economy.ledger, seller);
            #[unroll]
            for good in [Good::Bread as u32, Good::Coal as u32, Good::Wood as u32] {
                let relevant = output == good || dealer;
                if relevant {
                    let trip = economy.actors.distance(destination, seller);
                    let available = economy.ledger.number(Field::EnterpriseStatus, seller)
                        < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32)
                        && trip <= config.search_radius
                        && household_production::offers(&*economy, seller, good);
                    let terms = household_production::TradeFields::new(
                        &economy.actors,
                        config,
                        seller,
                        good,
                    );
                    let price = economy.ledger.number_at(terms.price, seller);
                    needs.bread = select(
                        available && price > 0 && good == comptime!(Good::Bread as u32),
                        min(needs.bread, price),
                        needs.bread,
                    );
                    needs.coal = select(
                        available && price > 0 && good == comptime!(Good::Coal as u32),
                        min(needs.coal, price),
                        needs.coal,
                    );
                    needs.wood = select(
                        available && price > 0 && good == comptime!(Good::Wood as u32),
                        min(needs.wood, price),
                        needs.wood,
                    );
                }
            }
        }
        needs
    }
}

#[cube]
pub(in crate::simulation) fn prepare_quotes(
    economy: &mut EconomyView<'_>,
    spatial: &[u32],
    quotes: &mut [u32],
    day: u32,
) {
    let config = economy.ledger.config;
    let member = ABSOLUTE_POS as u32;
    let cells = config.spatial_side * config.spatial_side;
    let segment = 2 * (cells + 1 + config.actors);
    if comptime!(config.home_mode == Feature::Enabled)
        && member < spatial[(segment + cells) as usize]
    {
        let destination = spatial[(segment + cells + 1 + member) as usize];
        let needs = LocalNeeds::new(economy, spatial, destination, day);
        let base = 4 * config.actors + 3 * destination;
        quotes[base as usize] = needs.bread;
        quotes[base as usize + 1] = needs.coal;
        quotes[base as usize + 2] = needs.wood;
    }
}

#[cube]
pub(in crate::simulation::economy) fn work_target(
    economy: &mut EconomyView<'_>,
    spatial: &[u32],
    quotes: &[u32],
    household: u32,
    day: u32,
    #[comptime] quote_mode: crate::simulation::enterprise::EntryQuoteMode,
) -> u32 {
    let config = economy.ledger.config;
    let cells = config.spatial_side * config.spatial_side;
    let segment = 2 * (cells + 1 + config.actors);
    let length = spatial[(segment + cells) as usize];
    let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
    let employer = economy.ledger.number(Field::Employer, household);
    let rural = economy.ledger.number(Field::LandPlots, household) > 0;
    let expected = divide(
        economy.ledger.value(Field::IncomeForecast, household),
        POLICY_SCALE,
    );
    let rent = times32(
        rent::household_quote(&economy.ledger, household),
        consumption::consumers(&economy.ledger, household),
    );
    let bread_budget = select(
        rural,
        economy.ledger.number(Field::FoodPrice, household),
        config.minimum_price,
    );
    let minimum_daily = plus(
        times32(
            economy.ledger.number(Field::NutritionNeed, household),
            bread_budget,
        ),
        rent,
    )
    .value;
    let nearby = employer > 0
        && economy.actors.distance(household, max(employer, 1) - 1) <= config.commute_speed;
    let scheduled = (day + household) % 30 == 0;
    let plots = economy.ledger.number(Field::LandPlots, household);
    let cultivation = CheckedTicks::ceiling_ratio(
        times32(plots, config.home.annual_yield_per_plot),
        config.home.cultivation_minutes,
        max(farming::CropCalendar::new(config).on(day).growing_days, 1),
    )
    .value;
    let remaining = positive(
        economy.ledger.value(Field::HouseholdBudget, household),
        economy.ledger.value(Field::Hours, household),
    );
    let displaced = lt(remaining, cultivation);
    let needed = select(rural, scheduled && displaced, !nearby);
    let sustainable = !rural || (employer > 0 && lt(minimum_daily, expected));
    let fee = select(nearby, 0u32, 100u32);
    let funded = !lt(economy.ledger.value(Field::Cash, household), ticks(fee, 0));
    let review = select(
        needed && sustainable && funded,
        select(
            employer > 0,
            comptime!(WorkReview::Employer as u32),
            comptime!(WorkReview::Vacancies as u32),
        ),
        comptime!(WorkReview::Inactive as u32),
    );
    let reviews = match review {
        0 => 0u32,
        1 => 1u32,
        _ => samples,
    };
    let mut target = 0u32;
    let mut earnings = zero();
    let mut nearest = u32::new(comptime!(i64::from(u32::MAX)));
    for sample in 0..reviews {
        let offset = sample * (length / samples) + sample * (length % samples) / samples;
        let firm = select(
            employer > 0,
            max(employer, 1) - 1,
            spatial[(segment + cells + 1 + (offset + day + household) % length) as usize],
        );
        let good = economy.actors.kind(firm).output();
        let shifts = match comptime!(config.cohorts) {
            Feature::Enabled => select(
                employer > 0,
                economy.ledger.number(Field::Employed, household),
                min(
                    employment::team(&economy.ledger, household),
                    economy.ledger.number(Field::Slots, firm),
                ),
            ),
            Feature::Disabled => u32::new(1),
        };
        let paid = times32(
            economy.ledger.number(Field::Wage, firm),
            economy.ledger.number(Field::LaborMinutes, firm) * shifts,
        );
        let trip = economy.actors.distance(household, firm);
        let hiring = select(
            employer > 0,
            employer == firm + 1,
            economy.ledger.number(Field::Slots, firm) > 0,
        );
        let urban = good > 0
            && good != comptime!(Good::Grain as u32)
            && good != comptime!(Good::Flax as u32)
            && good != comptime!(Good::Cotton as u32)
            && good != comptime!(Good::Dye as u32)
            && good != comptime!(Good::Housing as u32)
            && good != comptime!(Good::Railway as u32);
        let available = hiring
            && urban
            && (trip > config.commute_speed || review == comptime!(WorkReview::Employer as u32))
            && economy.ledger.number(Field::EnterpriseStatus, firm)
                == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32)
            && !lt(economy.ledger.value(Field::Cash, firm), paid);
        let preferable = lt(earnings, paid) || (equal(earnings, paid) && trip < nearest);
        if available && preferable {
            target = firm + 1;
            earnings = paid;
            nearest = trip;
        }
    }
    if target > 0 {
        let needs = match comptime!(quote_mode) {
            crate::simulation::enterprise::EntryQuoteMode::Cached => {
                LocalNeeds::cached(quotes, config, target - 1)
            }
            crate::simulation::enterprise::EntryQuoteMode::Sampled => {
                LocalNeeds::new(economy, spatial, target - 1, day)
            }
        };
        let heat = economy.ledger.number(Field::HeatingNeed, household);
        let food = times32(
            economy.ledger.number(Field::NutritionNeed, household),
            max(bread_budget, needs.bread),
        );
        let fuel = minimum(
            times32(ceildiv(heat, 2), needs.coal),
            times32(heat, needs.wood),
        );
        let rent = times32(
            rent::household_quote(&economy.ledger, target - 1),
            consumption::consumers(&economy.ledger, household),
        );
        let daily = plus(plus(food, fuel).value, rent).value;
        let travel = select(
            nearest > config.commute_speed,
            max(ceildiv(nearest, config.shipment_speed), 1),
            0u32,
        );
        let days = travel + config.coverage + config.ring;
        let fee = select(travel > 0, 100u32, 0u32);
        let budget = plus(times(daily, days).value, ticks(fee, 0)).value;
        let supplied = needs.bread <= config.maximum_price
            && (heat == 0 || min(needs.coal, needs.wood) <= config.maximum_price);
        let funded = !lt(economy.ledger.value(Field::Cash, household), budget);
        let income = select(rural, minimum(earnings, expected), earnings);
        target = select(supplied && funded && lt(daily, income), target, 0);
    }
    target
}

use super::*;

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

    fn new(
        state: &mut [u32],
        actors: &[u32],
        config: &EconomyConfig,
        spatial: &[u32],
        destination: u32,
        day: u32,
    ) -> Self {
        let cells = config.spatial_side * config.spatial_side;
        let segment = 2 * (cells + 1 + config.actors);
        let length = spatial[(segment + cells) as usize];
        let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
        let mut needs = LocalNeeds {
            bread: config.maximum_price + 1,
            coal: config.maximum_price + 1,
            wood: config.maximum_price + 1,
        };
        for sample in 0..samples {
            let offset = sample * (length / samples) + sample * (length % samples) / samples;
            let seller =
                spatial[(segment + cells + 1 + (offset + day + destination) % length) as usize];
            let good = kind(actors, seller).output();
            let relevant = good == comptime!(Good::Bread as u32)
                || good == comptime!(Good::Coal as u32)
                || good == comptime!(Good::Wood as u32);
            if relevant {
                let trip = distance(actors, destination, seller);
                let available = number(
                    state,
                    config,
                    comptime!(Field::EnterpriseStatus as u32),
                    seller,
                ) < comptime!(
                    crate::domain::enterprise::EnterpriseState::Liquidating as u32
                ) && trip <= config.search_radius
                    && household_production::offers(state, actors, config, seller, good);
                let price = number(state, config, comptime!(Field::Price as u32), seller);
                let days = max(ceildiv(trip, config.shipment_speed), 1);
                let delivered = price + days * config.delivery_penalty;
                needs.bread = select(
                    available && price > 0 && good == comptime!(Good::Bread as u32),
                    min(needs.bread, delivered),
                    needs.bread,
                );
                needs.coal = select(
                    available && price > 0 && good == comptime!(Good::Coal as u32),
                    min(needs.coal, delivered),
                    needs.coal,
                );
                needs.wood = select(
                    available && price > 0 && good == comptime!(Good::Wood as u32),
                    min(needs.wood, delivered),
                    needs.wood,
                );
            }
        }
        needs
    }
}

#[cube]
pub(in crate::simulation) fn prepare_quotes(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    quotes: &mut [u32],
    day: u32,
) {
    let member = ABSOLUTE_POS as u32;
    let cells = config.spatial_side * config.spatial_side;
    let segment = 2 * (cells + 1 + config.actors);
    if comptime!(config.home_mode == Feature::Enabled)
        && member < spatial[(segment + cells) as usize]
    {
        let destination = spatial[(segment + cells + 1 + member) as usize];
        let needs = LocalNeeds::new(state, actors, config, spatial, destination, day);
        let base = 4 * config.actors + 3 * destination;
        quotes[base as usize] = needs.bread;
        quotes[base as usize + 1] = needs.coal;
        quotes[base as usize + 2] = needs.wood;
    }
}

#[cube]
pub(super) fn work_target(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    spatial: &[u32],
    quotes: &[u32],
    household: u32,
    day: u32,
    #[comptime] quote_mode: crate::simulation::enterprise::EntryQuoteMode,
) -> u32 {
    let cells = config.spatial_side * config.spatial_side;
    let segment = 2 * (cells + 1 + config.actors);
    let length = spatial[(segment + cells) as usize];
    let samples = min(length, HOUSEHOLD_SEARCH_SAMPLES);
    let employer = number(state, config, comptime!(Field::Employer as u32), household);
    let rural = number(state, config, comptime!(Field::LandPlots as u32), household) > 0;
    let expected = divide(
        value(
            state,
            config,
            comptime!(Field::IncomeForecast as u32),
            household,
        ),
        POLICY_SCALE,
    );
    let minimum_daily = plus(
        times32(
            number(
                state,
                config,
                comptime!(Field::NutritionNeed as u32),
                household,
            ),
            config.minimum_price + config.delivery_penalty,
        ),
        ticks(config.household_rent, 0),
    )
    .value;
    let nearby =
        employer > 0 && distance(actors, household, max(employer, 1) - 1) <= config.commute_speed;
    let scheduled = (day + household) % 30 == 0;
    let needed = !nearby || (rural && scheduled);
    let sustainable = !rural || lt(minimum_daily, expected);
    let fee = select(nearby, 0u32, 100u32);
    let funded = !lt(
        value(state, config, comptime!(Field::Cash as u32), household),
        ticks(fee, 0),
    );
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
        let good = kind(actors, firm).output();
        let paid = times32(
            number(state, config, comptime!(Field::Wage as u32), firm),
            number(state, config, comptime!(Field::LaborMinutes as u32), firm),
        );
        let trip = distance(actors, household, firm);
        let hiring = select(
            employer > 0,
            employer == firm + 1,
            number(state, config, comptime!(Field::Slots as u32), firm) > 0,
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
            && number(
                state,
                config,
                comptime!(Field::EnterpriseStatus as u32),
                firm,
            ) == comptime!(crate::domain::enterprise::EnterpriseState::Operating as u32)
            && !lt(
                value(state, config, comptime!(Field::Cash as u32), firm),
                paid,
            );
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
                LocalNeeds::new(state, actors, config, spatial, target - 1, day)
            }
        };
        let heat = number(
            state,
            config,
            comptime!(Field::HeatingNeed as u32),
            household,
        );
        let food = times32(
            number(
                state,
                config,
                comptime!(Field::NutritionNeed as u32),
                household,
            ),
            needs.bread,
        );
        let fuel = minimum(
            times32(ceildiv(heat, 2), needs.coal),
            times32(heat, needs.wood),
        );
        let daily = plus(plus(food, fuel).value, ticks(config.household_rent, 0)).value;
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
        let funded = !lt(
            value(state, config, comptime!(Field::Cash as u32), household),
            budget,
        );
        let income = select(rural, minimum(earnings, expected), earnings);
        target = select(supplied && funded && lt(daily, income), target, 0);
    }
    target
}

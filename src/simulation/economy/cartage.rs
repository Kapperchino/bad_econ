use super::*;
use crate::simulation::{Device, configuration::Configuration};
use burn_wgpu::CubeTensor;

use crate::domain::railway::{CART_BILLABLE_MINUTES_PER_UNIT_DAY, CART_MINUTES_PER_UNIT_DAY};

#[cube]
pub(super) fn rate(config: &EconomyConfig, days: u32) -> u32 {
    days * comptime!(CART_BILLABLE_MINUTES_PER_UNIT_DAY) * config.starting_wage
}

#[cube]
pub(super) fn delivered_price(config: &EconomyConfig, quote: u32, trip: u32, days: u32) -> u32 {
    let animal_days = ceildiv(trip, config.shipment_speed);
    quote
        + select(
            trip > 0 && days >= animal_days,
            rate(config, animal_days),
            0,
        )
}

#[cube]
pub(super) fn waiting_field(config: &EconomyConfig, good: u32, days: u32) -> u32 {
    BASE_FIELDS as u32 + (Good::COUNT as u32 + good - 1) * config.ring + days
}

#[cube]
pub(super) fn booking_field(
    config: &EconomyConfig,
    good: u32,
    day: u32,
    days: u32,
    distance: u32,
) -> u32 {
    select(
        distance == 0,
        BASE_FIELDS as u32 + (good - 1) * config.ring + (day + days) % config.ring,
        waiting_field(config, good, days),
    )
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum CartageStage {
    Fleet,
    Household,
    Pay,
}

#[derive(CubeType, CubeLaunch)]
struct CartageInput {
    state: Box<[u32]>,
    actors: Box<[u32]>,
    drivers: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    day: u32,
    offset: u32,
    firms: u32,
    #[cube(comptime)]
    stage: CartageStage,
}

struct Cartage;
struct HaulGoods;

impl System for Cartage {
    type Input = CartageInput;
    type Passes = gpu_passes![HaulGoods];
}

#[cube]
fn collect(
    state: &mut [u32],
    actors: &[u32],
    config: &EconomyConfig,
    control: &[Atomic<u32>],
    driver: u32,
    client: u32,
    day: u32,
    #[comptime] time_field: Field,
) {
    let mut waiting = value(state, config, comptime!(Field::RoadFreight as u32), client);
    let time = number(state, config, comptime!(time_field as u32), driver);
    let travel = ceildiv(distance(actors, driver, client), config.commute_speed) * COMMUTE_LEGS;
    if !equal(waiting, zero())
        && time > travel
        && (comptime!(time_field == Field::CartageCapacity)
            || number(state, config, comptime!(Field::Worker as u32), driver) != NO_ACTOR)
    {
        let mut cash = value(state, config, comptime!(Field::Cash as u32), client);
        let mut minutes = time - travel;
        let mut income = zero();
        for days in 1..config.ring {
            let effort = days * comptime!(CART_MINUTES_PER_UNIT_DAY);
            let rate = rate(config, days);
            for good in 1..Good::COUNT as u32 + 1 {
                let field = waiting_field(config, good, days);
                let order = value(state, config, field, client);
                if !equal(order, zero()) && minutes >= effort {
                    let affordable = minimum(divide(cash, rate), ticks(minutes / effort, 0));
                    let load = minimum(order, affordable);
                    let fee = multiply(control, load, rate);
                    let slot =
                        BASE_FIELDS as u32 + (good - 1) * config.ring + (day + days) % config.ring;
                    increase(state, config, control, slot, client, load);
                    set(state, config, field, client, minus(order, load));
                    waiting = minus(waiting, load);
                    minutes -= load.extract(LOW_WORD) * effort;
                    cash = minus(cash, fee);
                    income = add_checked(control, income, fee);
                }
            }
        }
        if !equal(income, zero()) {
            let payable = value(
                state,
                config,
                comptime!(Field::CartagePayable as u32),
                client,
            );
            set(
                state,
                config,
                comptime!(Field::CartagePayable as u32),
                client,
                minus(payable, income),
            );
            set(state, config, comptime!(Field::Cash as u32), client, cash);
            set(
                state,
                config,
                comptime!(Field::RoadFreight as u32),
                client,
                waiting,
            );
            set32(state, config, comptime!(time_field as u32), driver, minutes);
            increase(
                state,
                config,
                control,
                comptime!(Field::CartageMinutes as u32),
                driver,
                ticks(time - minutes, 0),
            );
            increase(
                state,
                config,
                control,
                comptime!(Field::FreightEscrow as u32),
                driver,
                income,
            );
        }
    }
}

#[cube]
impl GpuPass<CartageInput> for HaulGoods {
    type Output = ();

    fn execute(mut input: CartageInput) {
        let index = ABSOLUTE_POS as u32;
        let start = match comptime!(input.stage) {
            CartageStage::Household => input.firms,
            _ => 0,
        };
        let count = match comptime!(input.stage) {
            CartageStage::Fleet => input.firms,
            _ => input.drivers.len() as u32 - start,
        };
        if index < count {
            let driver = input.drivers[(start + index) as usize];
            match comptime!(input.stage) {
                CartageStage::Fleet | CartageStage::Household => {
                    let time_field = comptime!(match input.stage {
                        CartageStage::Fleet => Field::CartageCapacity,
                        _ => Field::Time,
                    });
                    let slots = ceildiv(input.config.actors, count);
                    let first = (index + input.offset) % count;
                    let mut slot = 0;
                    while slot < slots
                        && number(
                            &mut input.state,
                            &input.config,
                            comptime!(time_field as u32),
                            driver,
                        ) > 0
                    {
                        let client = first + ((slot + input.day % slots) % slots) * count;
                        if client < input.config.actors && (client != driver || count == 1) {
                            collect(
                                &mut input.state,
                                &input.actors,
                                &input.config,
                                &input.control,
                                driver,
                                client,
                                input.day,
                                comptime!(time_field),
                            );
                        }
                        slot += 1;
                    }
                }
                CartageStage::Pay => {
                    let fee = value(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::FreightEscrow as u32),
                        driver,
                    );
                    increase(
                        &mut input.state,
                        &input.config,
                        &input.control,
                        comptime!(Field::Cash as u32),
                        driver,
                        fee,
                    );
                    set(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::CartageIncome as u32),
                        driver,
                        fee,
                    );
                    set32(
                        &mut input.state,
                        &input.config,
                        comptime!(Field::FreightEscrow as u32),
                        driver,
                        0,
                    );
                }
            }
        }
    }
}

pub(in crate::simulation) fn dispatch(
    device: &Device,
    state: &CubeTensor,
    actors: &CubeTensor,
    config: &Configuration,
    control: &CubeTensor,
    day: u32,
) {
    let carriers = config.cartage_accounts.meta.num_elements() as u32;
    let partners = carriers - config.cartage_firms;
    let fleet =
        (config.cartage_firms > 0).then_some((CartageStage::Fleet, day, config.cartage_firms));
    let households = (0..partners.min(4)).map(|round| {
        (
            CartageStage::Household,
            ((1 + u64::from(day) * 4 + u64::from(round)) % u64::from(partners)) as u32,
            partners,
        )
    });
    for (stage, offset, count) in
        fleet
            .into_iter()
            .chain(households)
            .chain(std::iter::once((CartageStage::Pay, 0, carriers)))
    {
        Cartage::launch_elements(
            device,
            count as usize,
            CartageInputLaunch {
                state: state.buffer_arg(),
                actors: actors.buffer_arg(),
                drivers: config.cartage_accounts.buffer_arg(),
                config: config.launch(),
                control: control.buffer_arg(),
                day,
                offset,
                firms: config.cartage_firms,
                stage,
            },
        );
    }
}

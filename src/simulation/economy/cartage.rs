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
    manifest: Box<[u32]>,
    config: EconomyConfig,
    control: Box<[Atomic<u32>]>,
    day: u32,
    offset: u32,
    firms: u32,
    #[cube(comptime)]
    stage: CartageStage,
    #[cube(comptime)]
    manifest_mode: Feature,
}

struct Cartage;
struct HaulGoods;

impl System for Cartage {
    type Input = CartageInput;
    type Passes = gpu_passes![HaulGoods];
}

#[cube(launch)]
fn index_pickups(state: &mut [u32], config: &EconomyConfig, manifest: &mut [u32]) {
    let ledger = Ledger::new(state, config);
    let client = ABSOLUTE_POS as u32;
    if client < config.actors {
        let waiting = ledger.value(Field::RoadFreight, client);
        let cash = ledger.value(Field::Cash, client);
        let mut days = 1u32;
        while days < config.ring
            && !equal(waiting, zero())
            && !lt(cash, ticks(rate(config, days), 0))
        {
            let mut goods = 0u32;
            for good in 1..Good::COUNT as u32 + 1 {
                let amount = ledger.value_at(waiting_field(config, good, days), client);
                goods |= u32::cast_from(!equal(amount, zero())) << (good - 1);
            }
            manifest[((days - 1) * config.actors + client) as usize] = goods;
            days += 1;
        }
    }
}

#[cube]
fn collect(
    economy: &mut EconomyView<'_>,
    control: &[Atomic<u32>],
    manifest: &[u32],
    #[comptime] manifest_mode: Feature,
    driver: u32,
    client: u32,
    day: u32,
    #[comptime] time_field: Field,
) {
    let config = economy.ledger.config;
    let mut waiting = economy.ledger.value(Field::RoadFreight, client);
    let time = economy.ledger.number(comptime!(time_field), driver);
    let travel = ceildiv(
        economy.actors.distance(driver, client),
        config.commute_speed,
    ) * COMMUTE_LEGS;
    if !equal(waiting, zero())
        && time > travel
        && (comptime!(time_field == Field::CartageCapacity)
            || economy.ledger.number(Field::Worker, driver) != NO_ACTOR)
    {
        let mut cash = economy.ledger.value(Field::Cash, client);
        let mut minutes = time - travel;
        let mut income = zero();
        let mut days = 1u32;
        while days < config.ring
            && !equal(waiting, zero())
            && minutes >= days * comptime!(CART_MINUTES_PER_UNIT_DAY)
            && !lt(cash, ticks(rate(config, days), 0))
        {
            let effort = days * comptime!(CART_MINUTES_PER_UNIT_DAY);
            let rate = rate(config, days);
            let mut goods = match comptime!(manifest_mode) {
                Feature::Enabled => manifest[((days - 1) * config.actors + client) as usize],
                Feature::Disabled => u32::new(comptime!((1i64 << Good::COUNT) - 1)),
            };
            while goods != 0
                && !equal(waiting, zero())
                && minutes >= effort
                && !lt(cash, ticks(rate, 0))
            {
                let good = goods.trailing_zeros() + 1;
                let field = waiting_field(config, good, days);
                let order = economy.ledger.value_at(field, client);
                if !equal(order, zero()) {
                    let affordable = minimum(divide(cash, rate), ticks(minutes / effort, 0));
                    let load = minimum(order, affordable);
                    let fee = multiply(control, load, rate);
                    let slot =
                        BASE_FIELDS as u32 + (good - 1) * config.ring + (day + days) % config.ring;
                    economy.ledger.increase_at(control, slot, client, load);
                    economy.ledger.set_at(field, client, minus(order, load));
                    waiting = minus(waiting, load);
                    minutes -= load.extract(LOW_WORD) * effort;
                    cash = minus(cash, fee);
                    income = add_checked(control, income, fee);
                }
                goods &= goods - 1;
            }
            days += 1;
        }
        if !equal(income, zero()) {
            let payable = economy.ledger.value(Field::CartagePayable, client);
            economy
                .ledger
                .set(Field::CartagePayable, client, minus(payable, income));
            economy.ledger.set(Field::Cash, client, cash);
            economy.ledger.set(Field::RoadFreight, client, waiting);
            economy
                .ledger
                .set_number(comptime!(time_field), driver, minutes);
            economy.ledger.increase(
                control,
                Field::CartageMinutes,
                driver,
                ticks(time - minutes, 0),
            );
            economy
                .ledger
                .increase(control, Field::FreightEscrow, driver, income);
        }
    }
}

#[cube]
impl GpuPass<CartageInput> for HaulGoods {
    type Output = ();

    fn execute(mut input: CartageInput) {
        let mut economy = EconomyView::new(&mut input.state, &input.actors, &input.config);
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
                    while slot < slots && economy.ledger.number(comptime!(time_field), driver) > 0 {
                        let client = first + ((slot + input.day % slots) % slots) * count;
                        if client < input.config.actors && (client != driver || count == 1) {
                            collect(
                                &mut economy,
                                &input.control,
                                &input.manifest,
                                comptime!(input.manifest_mode),
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
                    let fee = economy.ledger.value(Field::FreightEscrow, driver);
                    economy
                        .ledger
                        .increase(&input.control, Field::Cash, driver, fee);
                    economy.ledger.set(Field::CartageIncome, driver, fee);
                    economy.ledger.set_number(Field::FreightEscrow, driver, 0);
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
    manifest: &CubeTensor,
) {
    let carriers = config.cartage_accounts.meta.num_elements() as u32;
    let partners = carriers - config.cartage_firms;
    let manifest_mode = match (config.ring as usize - 1) * config.actors as usize
        <= manifest.meta.num_elements()
        && config.actors >= 4096
    {
        true => {
            index_pickups::launch(
                &device.client,
                device.count(config.actors as usize),
                CubeDim::new_1d(device.workgroup),
                state.buffer_arg(),
                config.launch(),
                manifest.buffer_arg(),
            );
            Feature::Enabled
        }
        false => Feature::Disabled,
    };
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
                manifest: manifest.buffer_arg(),
                manifest_mode,
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

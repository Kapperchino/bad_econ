use crate::simulation::economy::*;

#[derive(CubeType, CubeLaunch)]
pub(in crate::simulation) struct IndexInput {
    pub config: EconomyConfig,
    pub spatial: Box<[u32]>,
    pub scratch: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub parameters: EconomyParameters,
}

pub(in crate::simulation) struct MarketIndex;
pub(in crate::simulation) struct StoreOffsets;
pub(in crate::simulation) struct StoreMembers;

impl System for MarketIndex {
    type Input = IndexInput;
    type Passes = gpu_passes![StoreOffsets, StoreMembers];
}

#[cube]
impl GpuPass<IndexInput> for StoreOffsets {
    type Output = IndexInput;

    fn execute(mut input: IndexInput) -> IndexInput {
        input.parameters.refresh(&mut input.config);
        if input.control[LATCH_WORD].load() == 0 {
            index_offsets(
                &input.config,
                &mut input.spatial,
                &mut input.scratch,
                &input.parameters,
                ABSOLUTE_POS as u32,
            );
        }
        input
    }
}

#[cube]
impl GpuPass<IndexInput> for StoreMembers {
    type Output = ();

    fn execute(mut input: IndexInput) {
        if input.control[LATCH_WORD].load() == 0 {
            index_fill(
                &input.config,
                &mut input.spatial,
                &mut input.scratch,
                &input.parameters,
                ABSOLUTE_POS as u32,
            );
        }
    }
}

#[cube]
pub(super) fn discoverable(
    economy: &EconomyView<'_>,
    parameters: &EconomyParameters,
    firm: u32,
) -> bool {
    let market = comptime!(parameters.market);
    match market {
        MarketFlag::Labor => economy.ledger.number(Field::Slots, firm) > 0,
        MarketFlag::Input(_) | MarketFlag::Consumer(_) | MarketFlag::Household => {
            let good = parameters.traded_good();
            let distributor = economy.ledger.number(Field::EnterpriseModel, firm) == 1;
            let k = economy.actors.kind(firm);
            let status = economy.ledger.number(Field::EnterpriseStatus, firm);
            let producer = !k.is(comptime!(ActorKind::Household))
                && k.output() == good
                && status
                    < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32);
            producer
                || household_production::offers(economy, firm, good)
                || (distributor
                    && ((firm + parameters.day + good) % economy.ledger.config.retail_rotation
                        == 0
                        || !equal(
                            economy.ledger.value_at(
                                pricing::retail_field(
                                    economy.ledger.config,
                                    good,
                                    comptime!(crate::domain::RetailAccount::Demand),
                                ),
                                firm,
                            ),
                            zero(),
                        ))
                    && status
                        < comptime!(crate::domain::enterprise::EnterpriseState::Liquidating as u32))
        }
        MarketFlag::Payroll | MarketFlag::Distribution => false,
    }
}

#[cube]
pub(in crate::simulation::economy) fn index_count(
    economy: &EconomyView<'_>,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    member: u32,
) {
    let config = economy.ledger.config;
    let cells = config.spatial_side * config.spatial_side;
    let segment = index_segment(config, parameters);
    let members = spatial[(segment + cells) as usize];
    let mut eligible = 0u32;
    if member < members {
        let firm = spatial[(segment + cells + 1 + member) as usize];
        eligible = u32::cast_from(discoverable(economy, parameters, firm));
    }
    let prefix = parallel::scan_prefix(eligible, comptime!(parameters.scan));
    if member < config.actors {
        scratch[member as usize] = prefix.inclusive;
    }
    let block = CUBE_POS as u32;
    if UNIT_POS == 0 && block < ceildiv(config.actors, CUBE_DIM) {
        scratch[(config.actors + block) as usize] = prefix.total;
    }
}

#[cube]
fn index_segment(config: &EconomyConfig, parameters: &EconomyParameters) -> u32 {
    let cells = config.spatial_side * config.spatial_side;
    let good = parameters.traded_good();
    let households = good == Good::Bread as u32 && comptime!(config.home_mode == Feature::Disabled);
    (cells + 1 + config.actors) * select(households, 1u32, 2u32)
}

#[cube]
fn index_offset(config: &EconomyConfig, scratch: &mut [u32], members: u32) -> u32 {
    let mut offset = 0u32;
    if members > 0 {
        let member = members - 1;
        offset = scratch[member as usize];
        let block = member / CUBE_DIM;
        if block > 0 {
            offset += scratch[(config.actors + block - 1) as usize];
        }
    }
    offset
}

#[cube]
pub(in crate::simulation::economy) fn index_offsets(
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    cell: u32,
) {
    let cells = config.spatial_side * config.spatial_side;
    if cell <= cells {
        let segment = index_segment(config, parameters);
        let member = spatial[(segment + cell) as usize];
        spatial[cell as usize] = index_offset(config, scratch, member);
    }
}

#[cube]
pub(in crate::simulation::economy) fn index_fill(
    config: &EconomyConfig,
    spatial: &mut [u32],
    scratch: &mut [u32],
    parameters: &EconomyParameters,
    member: u32,
) {
    let cells = config.spatial_side * config.spatial_side;
    let segment = index_segment(config, parameters);
    let members = spatial[(segment + cells) as usize];
    if member < members {
        let mut previous = 0u32;
        if UNIT_POS > 0 {
            previous = scratch[member as usize - 1];
        }
        if scratch[member as usize] > previous {
            let offset = index_offset(config, scratch, member + 1);
            spatial[(cells + offset) as usize] = spatial[(segment + cells + 1 + member) as usize];
        }
    }
}

#[cube]
pub(in crate::simulation::economy) fn index_prices(
    economy: &EconomyView<'_>,
    spatial: &mut [u32],
    good: u32,
) {
    let ledger = &economy.ledger;
    let config = ledger.config;
    let cells = config.spatial_side * config.spatial_side;
    let count = spatial[cells as usize];
    let block = CUBE_POS as u32;
    let mut quote = u32::new(comptime!(i64::from(u32::MAX)));
    let mut member = block * DISCOVERY_BLOCK + UNIT_POS;
    while member < min((block + 1) * DISCOVERY_BLOCK, count) {
        let actor = spatial[(cells + 1 + member) as usize];
        let terms = household_production::TradeFields::new(&economy.actors, config, actor, good);
        quote = min(quote, ledger.number_at(terms.price, actor));
        member += CUBE_DIM;
    }
    let minimum = parallel::reduce::<u32>(quote, comptime!(Reduction::Minimum));
    if UNIT_POS == 0 && block * DISCOVERY_BLOCK < count {
        let base = 3 * (cells + 1 + config.actors);
        spatial[(base + block) as usize] = minimum;
    }
}

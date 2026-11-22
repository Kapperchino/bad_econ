use super::*;

#[cube]
fn regional_field(
    config: &EconomyConfig,
    region: u32,
    #[comptime] field: RailCapacityField,
) -> u32 {
    config.railway.region_base
        + comptime!(field as u32) * config.railway.region_blocks
        + region / config.railway.initial_actors
}

#[cube]
pub(super) fn regional_value(
    ledger: &Ledger<'_>,
    region: u32,
    #[comptime] field: RailCapacityField,
) -> Ticks {
    let config = ledger.config;
    ledger.value_at(
        regional_field(config, region, comptime!(field)),
        region % config.railway.initial_actors,
    )
}

#[cube]
pub(super) fn regional_number(
    ledger: &Ledger<'_>,
    region: u32,
    #[comptime] field: RailCapacityField,
) -> u32 {
    regional_value(ledger, region, comptime!(field)).extract(LOW_WORD)
}

#[cube]
pub(super) fn regional_set(
    ledger: &mut Ledger<'_>,
    region: u32,
    amount: Ticks,
    #[comptime] field: RailCapacityField,
) {
    let config = ledger.config;
    ledger.set_at(
        regional_field(config, region, comptime!(field)),
        region % config.railway.initial_actors,
        amount,
    );
}

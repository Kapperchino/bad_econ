use super::*;

#[derive(CubeType)]
pub(super) struct RegionalCapacity {
    first_field: u32,
    field_stride: u32,
    row: u32,
}

#[cube]
impl RegionalCapacity {
    pub(super) fn new(network: &RailNetwork, region: u32) -> Self {
        RegionalCapacity {
            first_field: network.region_base + region / network.initial_actors,
            field_stride: network.region_blocks,
            row: region % network.initial_actors,
        }
    }

    fn field(&self, #[comptime] field: RailCapacityField) -> u32 {
        self.first_field + comptime!(field as u32) * self.field_stride
    }

    pub(super) fn value(&self, ledger: &Ledger<'_>, #[comptime] field: RailCapacityField) -> Ticks {
        ledger.value_at(self.field(comptime!(field)), self.row)
    }

    pub(super) fn level(&self, ledger: &Ledger<'_>) -> u32 {
        self.value(ledger, comptime!(RailCapacityField::Level))
            .extract(LOW_WORD)
    }

    pub(super) fn set(
        &self,
        ledger: &mut Ledger<'_>,
        #[comptime] field: RailCapacityField,
        value: Ticks,
    ) {
        ledger.set_at(self.field(comptime!(field)), self.row, value);
    }
}

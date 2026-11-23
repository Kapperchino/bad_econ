use super::*;

#[derive(CubeType)]
pub(in crate::simulation::economy::market) struct SearchAxis {
    pub first: u32,
    pub last: u32,
    position: u32,
    radius: u32,
    #[cube(comptime)]
    single_cell: bool,
}

#[cube]
impl SearchAxis {
    pub(in crate::simulation::economy::market) fn around(
        position: u32,
        config: &EconomyConfig,
    ) -> Self {
        let center = position / config.search_radius;
        SearchAxis {
            first: max(center, 1u32) - 1,
            last: min(center + 1, config.spatial_side - 1),
            position,
            radius: config.search_radius,
            single_cell: false,
        }
    }

    pub(super) fn for_suppliers(
        position: u32,
        config: &EconomyConfig,
        #[comptime] suppliers: SupplierSearch,
    ) -> Self {
        match comptime!(suppliers) {
            SupplierSearch::SingleCell { .. } => SearchAxis {
                first: 0,
                last: 0,
                position,
                radius: config.search_radius,
                single_cell: true,
            },
            _ => Self::around(position, config),
        }
    }

    pub(in crate::simulation::economy::market) fn distance_to(&self, cell: u32) -> u32 {
        if comptime!(self.single_cell) {
            u32::cast_from(0u32)
        } else {
            max(self.position, cell * self.radius)
                - min(self.position, (cell + 1) * self.radius - 1)
        }
    }
}

use anyhow::Result;
use bad_econ::{
    domain::{
        Experiment, FirmSeed, Good, HouseholdSeed, Population, Position, ResourceMap,
        ResourceRegion, Scenario, ScenarioConfig, Technique,
    },
    simulation::analytics::{Coefficient, SparseMatrix},
};

pub(super) struct WoodFirms {
    pub seed: u64,
    pub firm_cash: u64,
    pub labor_minutes: u32,
}

impl WoodFirms {
    pub(super) fn configuration(self) -> Result<ScenarioConfig> {
        let mut config = Scenario::baseline(3, self.seed, Experiment::Concentrated)?
            .config()
            .clone();
        let position = Position { x: 0, y: 0 };
        config.distribution_rate = 0;
        config.forecast_weight = 0;
        config.coverage = 30;
        config.behavior.work_fatigue = 0;
        config.behavior.wage_increment = 0;
        config.resources = ResourceMap::Explicit {
            regions: vec![ResourceRegion::new(
                Good::Wood,
                position,
                Position {
                    x: config.map_size,
                    y: config.map_size,
                },
                config.map_size,
            )?],
        };
        config.population = Some(Population {
            households: (1..=3)
                .map(|identity| HouseholdSeed {
                    identity,
                    position,
                    cash: 0,
                    inventories: Good::food([0; 3]),
                    fatigue: 0,
                })
                .collect(),
            firms: (4..=6)
                .map(|identity| FirmSeed {
                    identity,
                    position,
                    owner: 1,
                    good: Good::Wood,
                    cash: self.firm_cash,
                    inventories: Good::food([0; 3]),
                    capacity: 1,
                    technique: Technique {
                        labor_minutes: self.labor_minutes,
                        input_ticks: 0,
                        output_ticks: 1,
                    },
                    wage: 1,
                    price: 1,
                })
                .collect(),
        });
        Ok(config)
    }
}

pub(super) fn dense_matrix(
    rows: usize,
    columns: usize,
    numerators: impl IntoIterator<Item = u32>,
    denominators: impl IntoIterator<Item = u32>,
) -> Result<SparseMatrix> {
    SparseMatrix::new(
        rows,
        columns,
        numerators
            .into_iter()
            .zip(denominators)
            .enumerate()
            .map(|(index, (numerator, denominator))| Coefficient {
                row: index / columns,
                column: index % columns,
                numerator,
                denominator,
            })
            .collect(),
    )
}

pub(super) fn uniform_matrix(
    rows: usize,
    columns: usize,
    numerators: impl IntoIterator<Item = u32>,
    denominator: u32,
) -> Result<SparseMatrix> {
    dense_matrix(rows, columns, numerators, std::iter::repeat(denominator))
}

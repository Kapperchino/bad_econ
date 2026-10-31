use super::*;

const MAX_MATRIX_CELLS: usize = 1_048_576;

pub struct DeviceMatrix {
    pub(super) data: CubeTensor,
    pub(super) rows: usize,
    pub(super) columns: usize,
    pub(super) scale: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct UnitFactor {
    numerator: u32,
    denominator: u32,
}

impl UnitFactor {
    pub fn new(numerator: u32, denominator: u32) -> Result<Self> {
        match numerator > 0 && denominator > 0 {
            true => Ok(Self {
                numerator,
                denominator,
            }),
            false => Err(anyhow!(
                "unit factors require positive numerator and denominator"
            )),
        }
    }

    pub fn inverse(self) -> Self {
        Self {
            numerator: self.denominator,
            denominator: self.numerator,
        }
    }
}

impl DeviceMatrix {
    pub(super) fn empty(device: &Device, rows: usize, columns: usize, scale: u32) -> Result<Self> {
        let cells = rows
            .checked_mul(columns)
            .context("analytical matrix size overflow")?;
        match cells <= MAX_MATRIX_CELLS
            && rows <= MAX_DIMENSION
            && columns <= MAX_DIMENSION
            && scale > 0
        {
            true => Ok(Self {
                data: device.empty((cells * INTERVAL_WORDS).max(INTERVAL_WORDS)),
                rows,
                columns,
                scale,
            }),
            false => Err(anyhow!(
                "interval matrix exceeds its capacity or has zero scale"
            )),
        }
    }

    pub fn new(device: &Device, matrix: &SparseMatrix, scale: u32) -> Result<Self> {
        let output = Self::empty(device, matrix.rows, matrix.columns, scale)?;
        let sparse = DeviceOperator::new(device, matrix);
        let control = device.tensor(&[0; CONTROL_WORDS]);
        kernels::dense_from_sparse::launch(
            &device.client,
            count(device, matrix.rows * matrix.columns),
            CubeDim::new_1d(device.workgroup),
            argument(&sparse.offsets),
            argument(&sparse.entries),
            argument(&output.data),
            argument(&control),
            matrix.rows as u32,
            matrix.columns as u32,
            scale,
        );
        faults(device, &control)?;
        Ok(output)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn scale(&self) -> u32 {
        self.scale
    }

    pub fn read(&self, device: &Device) -> Result<Vec<Interval>> {
        DeviceVector {
            data: self.data.clone(),
            length: self.rows * self.columns,
        }
        .read(device)
    }

    pub fn apply(&self, device: &Device, vector: &DeviceVector) -> Result<DeviceVector> {
        let right = Self {
            data: vector.data.clone(),
            rows: vector.length,
            columns: 1,
            scale: self.scale,
        };
        let product = self.product(device, &right)?;
        Ok(DeviceVector {
            data: product.data,
            length: product.rows,
        })
    }

    pub fn product(&self, device: &Device, right: &Self) -> Result<Self> {
        match self.columns == right.rows && self.scale == right.scale {
            false => Err(anyhow!("matrix product dimensions or scales disagree")),
            true => {
                let output = Self::empty(device, self.rows, right.columns, self.scale)?;
                let control = device.tensor(&[0; CONTROL_WORDS]);
                kernels::dense_product::launch(
                    &device.client,
                    count(device, output.rows * output.columns),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.data),
                    argument(&right.data),
                    argument(&output.data),
                    argument(&control),
                    self.rows as u32,
                    self.columns as u32,
                    right.columns as u32,
                    self.scale,
                );
                faults(device, &control)?;
                Ok(output)
            }
        }
    }

    pub fn add(&self, device: &Device, right: &Self) -> Result<Self> {
        match self.rows == right.rows && self.columns == right.columns && self.scale == right.scale
        {
            false => Err(anyhow!("matrix sum dimensions or scales disagree")),
            true => {
                let left = DeviceVector {
                    data: self.data.clone(),
                    length: self.rows * self.columns,
                };
                let right = DeviceVector {
                    data: right.data.clone(),
                    length: right.rows * right.columns,
                };
                Ok(Self {
                    data: left.add(device, &right)?.data,
                    rows: self.rows,
                    columns: self.columns,
                    scale: self.scale,
                })
            }
        }
    }

    pub fn transpose(&self, device: &Device) -> Result<Self> {
        let output = Self::empty(device, self.columns, self.rows, self.scale)?;
        kernels::dense_transpose::launch(
            &device.client,
            count(device, self.rows * self.columns),
            CubeDim::new_1d(device.workgroup),
            argument(&self.data),
            argument(&output.data),
            self.rows as u32,
            self.columns as u32,
        );
        Ok(output)
    }

    pub fn scale_units(
        &self,
        device: &Device,
        rows: &[UnitFactor],
        columns: &[UnitFactor],
    ) -> Result<Self> {
        match rows.len() == self.rows && columns.len() == self.columns {
            false => Err(anyhow!("unit factors do not match matrix dimensions")),
            true => {
                let encode = |factors: &[UnitFactor]| {
                    padded(
                        factors
                            .iter()
                            .flat_map(|factor| [factor.numerator, factor.denominator])
                            .collect(),
                    )
                };
                let row_factors = device.tensor(&encode(rows));
                let column_factors = device.tensor(&encode(columns));
                let output = Self::empty(device, self.rows, self.columns, self.scale)?;
                let control = device.tensor(&[0; CONTROL_WORDS]);
                kernels::dense_units::launch(
                    &device.client,
                    count(device, self.rows * self.columns),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.data),
                    argument(&row_factors),
                    argument(&column_factors),
                    argument(&output.data),
                    argument(&control),
                    self.rows as u32,
                    self.columns as u32,
                );
                faults(device, &control)?;
                Ok(output)
            }
        }
    }

    pub fn per_hour_goods(
        device: &Device,
        subsistence: &Self,
        education: &Self,
        lifetimes: &[u32],
    ) -> Result<Self> {
        subsistence.add(device, &education.per_hour_training(device, lifetimes)?)
    }

    pub fn per_hour_training(&self, device: &Device, lifetimes: &[u32]) -> Result<Self> {
        let rows = vec![UnitFactor::new(1, 1)?; self.rows];
        let columns = lifetimes
            .iter()
            .map(|&lifetime| UnitFactor::new(1, lifetime))
            .collect::<Result<Vec<_>>>()?;
        self.scale_units(device, &rows, &columns)
    }
}

impl DeviceOperator {
    pub fn inverse(
        &self,
        device: &Device,
        scale: u32,
        certificate: &Contraction,
        budget: SolveBudget,
    ) -> Result<Diagnostic<DeviceMatrix>> {
        let output = DeviceMatrix::empty(device, self.rows, self.columns, scale)?;
        let mut result = Diagnostic::Available(());
        for column in 0..self.columns {
            let mut rhs = vec![0; self.rows];
            *rhs.get_mut(column)
                .context("inverse requires a square matrix")? = u64::from(scale);
            let solution = self.solve(
                device,
                &DeviceVector::exact(device, &rhs)?,
                certificate,
                budget,
            )?;
            match solution.values(device)? {
                Diagnostic::Available(values) => kernels::dense_column::launch(
                    &device.client,
                    count(device, self.rows),
                    CubeDim::new_1d(device.workgroup),
                    argument(&values.data),
                    argument(&output.data),
                    self.rows as u32,
                    self.columns as u32,
                    column as u32,
                ),
                Diagnostic::Inadmissible => result = Diagnostic::Inadmissible,
                _ => result = Diagnostic::UnresolvedAccuracy,
            }
        }
        Ok(match result {
            Diagnostic::Available(()) => Diagnostic::Available(output),
            Diagnostic::Inadmissible => Diagnostic::Inadmissible,
            _ => Diagnostic::UnresolvedAccuracy,
        })
    }
}

pub struct BoundaryBlocks<'a> {
    pub retained_goods: &'a DeviceMatrix,
    pub retained_from_outside: &'a DeviceMatrix,
    pub outside_from_retained: &'a DeviceMatrix,
    pub outside_inverse: &'a DeviceMatrix,
    pub retained_labor: &'a DeviceMatrix,
    pub outside_labor: &'a DeviceMatrix,
    pub retained_reproduction: &'a DeviceMatrix,
    pub outside_reproduction: &'a DeviceMatrix,
    pub training: &'a DeviceMatrix,
}

pub struct BoundaryReduction {
    pub goods: DeviceMatrix,
    pub labor: DeviceMatrix,
    pub reproduction: DeviceMatrix,
    pub training: DeviceMatrix,
}

impl BoundaryReduction {
    pub fn new(device: &Device, blocks: BoundaryBlocks<'_>) -> Result<Self> {
        let outside_goods = blocks
            .outside_inverse
            .product(device, blocks.outside_from_retained)?;
        let outside_reproduction = blocks
            .outside_inverse
            .product(device, blocks.outside_reproduction)?;
        Ok(Self {
            goods: blocks.retained_goods.add(
                device,
                &blocks
                    .retained_from_outside
                    .product(device, &outside_goods)?,
            )?,
            labor: blocks.retained_labor.add(
                device,
                &blocks.outside_labor.product(device, &outside_goods)?,
            )?,
            reproduction: blocks.retained_reproduction.add(
                device,
                &blocks
                    .retained_from_outside
                    .product(device, &outside_reproduction)?,
            )?,
            training: blocks.training.add(
                device,
                &blocks
                    .outside_labor
                    .product(device, &outside_reproduction)?,
            )?,
        })
    }

    pub fn effective_demand(
        device: &Device,
        retained: &DeviceVector,
        outside: &DeviceVector,
        outside_inverse: &DeviceMatrix,
        retained_from_outside: &DeviceMatrix,
    ) -> Result<DeviceVector> {
        retained.add(
            device,
            &retained_from_outside.apply(device, &outside_inverse.apply(device, outside)?)?,
        )
    }
}

pub struct DeviceSignedVector {
    pub(super) positive: DeviceVector,
    pub(super) negative: DeviceVector,
}

impl DeviceSignedVector {
    pub fn difference(positive: DeviceVector, negative: DeviceVector) -> Result<Self> {
        match positive.length == negative.length {
            true => Ok(Self { positive, negative }),
            false => Err(anyhow!("signed vector dimensions disagree")),
        }
    }

    pub fn apply(&self, device: &Device, matrix: &DeviceMatrix) -> Result<Self> {
        Self::difference(
            matrix.apply(device, &self.positive)?,
            matrix.apply(device, &self.negative)?,
        )
    }

    pub fn read(&self, device: &Device) -> Result<Vec<SignedInterval>> {
        let output = device.empty((self.positive.length * 6).max(6));
        kernels::signed_vector::launch(
            &device.client,
            count(device, self.positive.length),
            CubeDim::new_1d(device.workgroup),
            argument(&self.positive.data),
            argument(&self.negative.data),
            argument(&output),
            self.positive.length as u32,
        );
        Ok(device
            .read(&output)?
            .chunks_exact(6)
            .take(self.positive.length)
            .map(|words| SignedInterval {
                lower: SignedTicks {
                    sign: match words[0] {
                        0 => Sign::NonNegative,
                        _ => Sign::Negative,
                    },
                    magnitude: decode(&words[1..3]),
                },
                upper: SignedTicks {
                    sign: match words[3] {
                        0 => Sign::NonNegative,
                        _ => Sign::Negative,
                    },
                    magnitude: decode(&words[4..6]),
                },
            })
            .collect())
    }
}

pub struct InventoryFlows<'a> {
    pub opening: &'a DeviceVector,
    pub closing: &'a DeviceVector,
    pub production: &'a DeviceVector,
    pub final_demand: &'a DeviceVector,
    pub research_inputs: &'a DeviceVector,
    pub direct_research_hours: &'a DeviceVector,
}

pub struct InventoryAccount {
    pub physical_residual: DeviceSignedVector,
    pub actual_hours: DeviceVector,
    pub reconciled_hours: DeviceSignedVector,
}

impl InventoryAccount {
    pub fn new(
        device: &Device,
        goods: &DeviceOperator,
        direct_labor: &DeviceOperator,
        labor_requirements: &DeviceMatrix,
        flows: InventoryFlows<'_>,
    ) -> Result<Self> {
        let material_uses = goods
            .apply(device, flows.production)?
            .add(device, flows.final_demand)?
            .add(device, flows.research_inputs)?
            .add(device, flows.closing)?;
        let sources = flows.opening.add(device, flows.production)?;
        let requirement = flows
            .final_demand
            .add(device, flows.research_inputs)?
            .add(device, flows.closing)?;
        Ok(Self {
            physical_residual: DeviceSignedVector::difference(sources, material_uses)?,
            actual_hours: direct_labor
                .apply(device, flows.production)?
                .add(device, flows.direct_research_hours)?,
            reconciled_hours: DeviceSignedVector::difference(
                labor_requirements
                    .apply(device, &requirement)?
                    .add(device, flows.direct_research_hours)?,
                labor_requirements.apply(device, flows.opening)?,
            )?,
        })
    }
}

mod accounts;
mod kernels;
mod matrices;
mod request;
mod retained;
pub use retained::*;
mod sensitivity;
pub use accounts::*;
pub use matrices::*;
pub use request::*;
pub use sensitivity::*;

use super::{
    Device,
    flags::FaultFlags,
    layout::{TICK_WORDS, WORD_BYTES},
};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use cubecl::{CubeCount, CubeDim, prelude::BufferArg};
use serde::{Deserialize, Serialize};

pub const ANALYTICAL_PROFILE: &str = "DirectedRationalTicksV1";
const INTERVAL_WORDS: usize = 4;
const CONTROL_WORDS: usize = 4;
const MAX_DIMENSION: usize = 65_535;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Coefficient {
    pub row: usize,
    pub column: usize,
    pub numerator: u32,
    pub denominator: u32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "MatrixDefinition", into = "MatrixDefinition")]
pub struct SparseMatrix {
    rows: usize,
    columns: usize,
    entries: Vec<Coefficient>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MatrixDefinition {
    rows: usize,
    columns: usize,
    entries: Vec<Coefficient>,
}
impl TryFrom<MatrixDefinition> for SparseMatrix {
    type Error = anyhow::Error;
    fn try_from(value: MatrixDefinition) -> Result<Self> {
        Self::new(value.rows, value.columns, value.entries)
    }
}
impl From<SparseMatrix> for MatrixDefinition {
    fn from(value: SparseMatrix) -> Self {
        Self {
            rows: value.rows,
            columns: value.columns,
            entries: value.entries,
        }
    }
}

impl SparseMatrix {
    pub fn new(rows: usize, columns: usize, mut entries: Vec<Coefficient>) -> Result<Self> {
        let admitted = rows <= MAX_DIMENSION
            && columns <= MAX_DIMENSION
            && entries.len() <= u32::MAX as usize / kernels::ENTRY_WORDS
            && entries
                .iter()
                .all(|entry| entry.row < rows && entry.column < columns && entry.denominator > 0);
        match admitted {
            false => Err(anyhow!(
                "invalid sparse matrix shape, coordinate, denominator, or capacity"
            )),
            true => {
                entries.sort_by_key(|entry| (entry.row, entry.column));
                match entries
                    .windows(2)
                    .all(|pair| (pair[0].row, pair[0].column) != (pair[1].row, pair[1].column))
                {
                    true => Ok(Self {
                        rows,
                        columns,
                        entries,
                    }),
                    false => Err(anyhow!("sparse matrix has duplicate coordinates")),
                }
            }
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn entries(&self) -> &[Coefficient] {
        &self.entries
    }

    pub fn transpose(&self) -> Self {
        let mut entries = self
            .entries
            .iter()
            .map(|entry| Coefficient {
                row: entry.column,
                column: entry.row,
                ..*entry
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| (entry.row, entry.column));
        Self {
            rows: self.columns,
            columns: self.rows,
            entries,
        }
    }

    pub fn coupled(
        goods: &Self,
        labor: &Self,
        reproduction: &Self,
        training: &Self,
    ) -> Result<Self> {
        let n = goods.rows;
        let m = labor.rows;
        let shapes = goods.columns == n
            && labor.columns == n
            && reproduction.rows == n
            && reproduction.columns == m
            && training.rows == m
            && training.columns == m;
        match shapes {
            false => Err(anyhow!("coupled account block dimensions disagree")),
            true => {
                let entries = goods
                    .entries
                    .iter()
                    .map(|entry| Coefficient {
                        row: entry.column,
                        column: entry.row,
                        ..*entry
                    })
                    .chain(labor.entries.iter().map(|entry| Coefficient {
                        row: entry.column,
                        column: n + entry.row,
                        ..*entry
                    }))
                    .chain(reproduction.entries.iter().map(|entry| Coefficient {
                        row: n + entry.column,
                        column: entry.row,
                        ..*entry
                    }))
                    .chain(training.entries.iter().map(|entry| Coefficient {
                        row: n + entry.column,
                        column: n + entry.row,
                        ..*entry
                    }))
                    .collect();
                Self::new(n + m, n + m, entries)
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextVersion {
    pub techniques: u64,
    pub units: u64,
    pub classification: u64,
    pub access: u64,
    pub reproduction: u64,
    pub occupations: u64,
    pub reference_occupation: u64,
    pub research_catalogue: u64,
    pub capability_graph: u64,
    pub numerical_profile: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Diagnostic<T> {
    Available(T),
    MissingData,
    Inapplicable,
    Inadmissible,
    UnresolvedAccuracy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Interval {
    pub lower: u64,
    pub upper: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CertifiedValues {
    pub values: Vec<Interval>,
    pub weighted_error: u64,
    pub iterations: u32,
    pub profile: &'static str,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(try_from = "ContractionDefinition", into = "ContractionDefinition")]
pub struct Contraction {
    numerator: u32,
    denominator: u32,
    weights: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractionDefinition {
    numerator: u32,
    denominator: u32,
    weights: Vec<u32>,
}
impl TryFrom<ContractionDefinition> for Contraction {
    type Error = anyhow::Error;
    fn try_from(value: ContractionDefinition) -> Result<Self> {
        Self::new(value.numerator, value.denominator, value.weights)
    }
}
impl From<Contraction> for ContractionDefinition {
    fn from(value: Contraction) -> Self {
        Self {
            numerator: value.numerator,
            denominator: value.denominator,
            weights: value.weights,
        }
    }
}

impl Contraction {
    pub fn new(numerator: u32, denominator: u32, weights: Vec<u32>) -> Result<Self> {
        match numerator < denominator
            && !weights.is_empty()
            && weights.iter().all(|&weight| weight > 0)
        {
            true => Ok(Self {
                numerator,
                denominator,
                weights,
            }),
            false => Err(anyhow!(
                "contraction requires positive weights and 0 <= numerator < denominator"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolveBudget {
    pub iterations: u32,
    pub weighted_tolerance: u64,
}

pub struct DeviceVector {
    data: CubeTensor,
    length: usize,
}

impl DeviceVector {
    pub fn exact(device: &Device, values: &[u64]) -> Result<Self> {
        match values.len() <= MAX_DIMENSION {
            true => {
                let words = values
                    .iter()
                    .flat_map(|&value| {
                        [
                            value as u32,
                            (value >> 32) as u32,
                            value as u32,
                            (value >> 32) as u32,
                        ]
                    })
                    .collect::<Vec<_>>();
                Ok(Self {
                    data: device.tensor(&padded(words)),
                    length: values.len(),
                })
            }
            false => Err(anyhow!("analytical vector exceeds dimension limit")),
        }
    }

    fn empty(device: &Device, length: usize) -> Self {
        Self {
            data: device.empty((length * INTERVAL_WORDS).max(INTERVAL_WORDS)),
            length,
        }
    }

    pub fn len(&self) -> usize {
        self.length
    }
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    pub fn read(&self, device: &Device) -> Result<Vec<Interval>> {
        Ok(device
            .read(&self.data)?
            .chunks_exact(INTERVAL_WORDS)
            .take(self.length)
            .map(|words| Interval {
                lower: decode(&words[..TICK_WORDS]),
                upper: decode(&words[TICK_WORDS..]),
            })
            .collect())
    }

    pub fn add(&self, device: &Device, other: &Self) -> Result<Self> {
        match self.length == other.length {
            false => Err(anyhow!("analytical vector dimensions disagree")),
            true => {
                let output = Self::empty(device, self.length);
                let control = device.tensor(&[0; CONTROL_WORDS]);
                kernels::vector_add::launch(
                    &device.client,
                    count(device, self.length),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.data),
                    argument(&other.data),
                    argument(&output.data),
                    argument(&control),
                    self.length as u32,
                );
                faults(device, &control)?;
                Ok(output)
            }
        }
    }

    pub fn dot(&self, device: &Device, other: &Self, scale: u32) -> Result<Self> {
        match self.length == other.length && scale > 0 {
            false => Err(anyhow!(
                "dot product requires equal dimensions and positive scale"
            )),
            true => {
                let output = Self::empty(device, 1);
                let control = device.tensor(&[0; CONTROL_WORDS]);
                kernels::vector_dot::launch(
                    &device.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.data),
                    argument(&other.data),
                    argument(&output.data),
                    argument(&control),
                    self.length as u32,
                    scale,
                );
                faults(device, &control)?;
                Ok(output)
            }
        }
    }

    pub fn normalize(
        &self,
        device: &Device,
        reference: usize,
        scale: u32,
    ) -> Result<Diagnostic<Self>> {
        match reference < self.length && scale > 0 {
            false => Err(anyhow!(
                "normalization requires an existing reference and positive scale"
            )),
            true => {
                let output = Self::empty(device, self.length);
                let control = device.tensor(&[0; CONTROL_WORDS]);
                kernels::normalize::launch(
                    &device.client,
                    count(device, self.length),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.data),
                    argument(&output.data),
                    argument(&control),
                    self.length as u32,
                    reference as u32,
                    scale,
                );
                let status = faults(device, &control)?;
                Ok(match status[0] {
                    0 => Diagnostic::Available(output),
                    _ => Diagnostic::UnresolvedAccuracy,
                })
            }
        }
    }
}

pub struct DeviceOperator {
    offsets: CubeTensor,
    entries: CubeTensor,
    rows: usize,
    columns: usize,
}

impl DeviceOperator {
    pub fn new(device: &Device, matrix: &SparseMatrix) -> Self {
        let mut offsets = vec![0; matrix.rows + 1];
        matrix
            .entries
            .iter()
            .for_each(|entry| offsets[entry.row + 1] += 1);
        (1..offsets.len()).for_each(|index| offsets[index] += offsets[index - 1]);
        let entries = matrix
            .entries
            .iter()
            .flat_map(|entry| [entry.column as u32, entry.numerator, entry.denominator])
            .collect::<Vec<_>>();
        Self {
            offsets: device.tensor(&padded(offsets)),
            entries: device.tensor(&padded(entries)),
            rows: matrix.rows,
            columns: matrix.columns,
        }
    }

    pub fn apply(&self, device: &Device, input: &DeviceVector) -> Result<DeviceVector> {
        match self.columns == input.length {
            false => Err(anyhow!("matrix and vector dimensions disagree")),
            true => {
                let output = DeviceVector::empty(device, self.rows);
                let control = device.tensor(&[0; CONTROL_WORDS]);
                kernels::matvec::launch(
                    &device.client,
                    count(device, self.rows),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.offsets),
                    argument(&self.entries),
                    argument(&input.data),
                    argument(&output.data),
                    argument(&control),
                    self.rows as u32,
                );
                faults(device, &control)?;
                Ok(output)
            }
        }
    }

    pub fn step(
        &self,
        device: &Device,
        rhs: &DeviceVector,
        previous: &DeviceVector,
    ) -> Result<DeviceVector> {
        self.apply(device, previous)?.add(device, rhs)
    }

    pub fn solve(
        &self,
        device: &Device,
        rhs: &DeviceVector,
        certificate: &Contraction,
        budget: SolveBudget,
    ) -> Result<DeviceSolution> {
        let initial = DeviceVector::exact(device, &vec![0; self.rows])?;
        self.solve_from(device, rhs, &initial, certificate, budget)
    }

    pub fn solve_from(
        &self,
        device: &Device,
        rhs: &DeviceVector,
        initial: &DeviceVector,
        certificate: &Contraction,
        budget: SolveBudget,
    ) -> Result<DeviceSolution> {
        let admitted = self.rows == self.columns
            && rhs.length == self.rows
            && initial.length == self.rows
            && certificate.weights.len() == self.rows
            && self.rows > 0
            && budget.iterations <= 100_000;
        match admitted {
            false => Err(anyhow!(
                "certified solve dimensions or iteration budget are invalid"
            )),
            true => {
                let weights = device.tensor(&certificate.weights);
                let control = device.tensor(&[0; CONTROL_WORDS]);
                let rows = device.empty(self.rows * 2);
                kernels::certificate_rows::launch(
                    &device.client,
                    count(device, self.rows),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.offsets),
                    argument(&self.entries),
                    argument(&weights),
                    argument(&rows),
                    argument(&control),
                    self.rows as u32,
                    certificate.numerator,
                    certificate.denominator,
                );
                kernels::certificate_reduce::launch(
                    &device.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(device.workgroup),
                    argument(&rows),
                    argument(&control),
                    self.rows as u32,
                );
                let mut previous = DeviceVector {
                    data: initial.data.clone(),
                    length: initial.length,
                };
                let mut next = DeviceVector::empty(device, self.rows);
                for _ in 0..budget.iterations {
                    kernels::iterate::launch(
                        &device.client,
                        count(device, self.rows),
                        CubeDim::new_1d(device.workgroup),
                        argument(&self.offsets),
                        argument(&self.entries),
                        argument(&rhs.data),
                        argument(&previous.data),
                        argument(&next.data),
                        argument(&control),
                        self.rows as u32,
                    );
                    std::mem::swap(&mut previous, &mut next);
                }
                let residuals = device.empty(self.rows * TICK_WORDS);
                kernels::residual::launch(
                    &device.client,
                    count(device, self.rows),
                    CubeDim::new_1d(device.workgroup),
                    argument(&self.offsets),
                    argument(&self.entries),
                    argument(&rhs.data),
                    argument(&previous.data),
                    argument(&weights),
                    argument(&residuals),
                    argument(&control),
                    self.rows as u32,
                );
                let error = device.empty(TICK_WORDS);
                kernels::error_bound::launch(
                    &device.client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(device.workgroup),
                    argument(&residuals),
                    argument(&error),
                    argument(&control),
                    self.rows as u32,
                    certificate.numerator,
                    certificate.denominator,
                    budget.weighted_tolerance as u32,
                    (budget.weighted_tolerance >> 32) as u32,
                );
                let output = DeviceVector::empty(device, self.rows);
                kernels::solution_bounds::launch(
                    &device.client,
                    count(device, self.rows),
                    CubeDim::new_1d(device.workgroup),
                    argument(&previous.data),
                    argument(&weights),
                    argument(&error),
                    argument(&output.data),
                    argument(&control),
                    self.rows as u32,
                );
                faults(device, &control)?;
                Ok(DeviceSolution {
                    values: output,
                    control,
                    error,
                    iterations: budget.iterations,
                })
            }
        }
    }
}

pub struct DeviceSolution {
    values: DeviceVector,
    control: CubeTensor,
    error: CubeTensor,
    iterations: u32,
}

impl DeviceSolution {
    pub fn values(&self, device: &Device) -> Result<Diagnostic<&DeviceVector>> {
        let status = faults(device, &self.control)?;
        Ok(match status[0] {
            kernels::AVAILABLE => Diagnostic::Available(&self.values),
            kernels::INADMISSIBLE => Diagnostic::Inadmissible,
            _ => Diagnostic::UnresolvedAccuracy,
        })
    }

    pub fn read(&self, device: &Device) -> Result<Diagnostic<CertifiedValues>> {
        Ok(match self.values(device)? {
            Diagnostic::Available(values) => Diagnostic::Available(CertifiedValues {
                values: values.read(device)?,
                weighted_error: decode(&device.read(&self.error)?),
                iterations: self.iterations,
                profile: ANALYTICAL_PROFILE,
            }),
            Diagnostic::Inadmissible => Diagnostic::Inadmissible,
            _ => Diagnostic::UnresolvedAccuracy,
        })
    }
}

pub struct CachedAccount<T> {
    version: ContextVersion,
    value: T,
}

impl<T> CachedAccount<T> {
    pub fn new(version: ContextVersion, value: T) -> Self {
        Self { version, value }
    }
    pub fn get(&self, current: ContextVersion) -> Option<&T> {
        (self.version == current).then_some(&self.value)
    }
}

fn padded(mut words: Vec<u32>) -> Vec<u32> {
    words.resize(words.len().max(INTERVAL_WORDS), 0);
    words
}

fn argument(tensor: &CubeTensor) -> BufferArg {
    unsafe { BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0]) }
}

fn count(device: &Device, items: usize) -> CubeCount {
    CubeCount::Static((items as u32).div_ceil(device.workgroup).max(1), 1, 1)
}

fn faults(device: &Device, control: &CubeTensor) -> Result<Vec<u32>> {
    let words = device.read(control)?;
    FaultFlags::new(*words.get(1).context("missing analytical fault word")?)?.into_result()?;
    Ok(words)
}

fn decode(words: &[u32]) -> u64 {
    u64::from(words[0]) | (u64::from(words[1]) << (WORD_BYTES * 8))
}

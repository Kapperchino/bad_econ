use super::*;
use crate::simulation::{pass::GpuPass, system::gpu_passes};
use cubecl::prelude::*;

pub struct DeviceSignedMatrix {
    positive: DeviceMatrix,
    negative: DeviceMatrix,
}

impl DeviceSignedMatrix {
    pub fn difference(positive: DeviceMatrix, negative: DeviceMatrix) -> Result<Self> {
        match positive.rows == negative.rows
            && positive.columns == negative.columns
            && positive.scale == negative.scale
        {
            true => Ok(Self { positive, negative }),
            false => Err(anyhow!("signed matrix dimensions or scales disagree")),
        }
    }

    pub fn rows(&self) -> usize {
        self.positive.rows
    }

    pub fn columns(&self) -> usize {
        self.positive.columns
    }

    pub fn scale(&self) -> u32 {
        self.positive.scale
    }

    pub fn read(&self, device: &Device) -> Result<Vec<SignedInterval>> {
        DeviceSignedVector::difference(
            DeviceVector {
                data: self.positive.data.clone(),
                length: self.rows() * self.columns(),
            },
            DeviceVector {
                data: self.negative.data.clone(),
                length: self.rows() * self.columns(),
            },
        )?
        .read(device)
    }

    pub fn add(&self, device: &Device, right: &Self) -> Result<Self> {
        Self::difference(
            self.positive.add(device, &right.positive)?,
            self.negative.add(device, &right.negative)?,
        )
    }

    pub fn left_product(&self, device: &Device, left: &DeviceMatrix) -> Result<Self> {
        Self::difference(
            left.product(device, &self.positive)?,
            left.product(device, &self.negative)?,
        )
    }

    pub fn right_product(&self, device: &Device, right: &DeviceMatrix) -> Result<Self> {
        Self::difference(
            self.positive.product(device, right)?,
            self.negative.product(device, right)?,
        )
    }

    pub fn transpose(&self, device: &Device) -> Result<Self> {
        Self::difference(
            self.positive.transpose(device)?,
            self.negative.transpose(device)?,
        )
    }

    pub fn apply(&self, device: &Device, vector: &DeviceVector) -> Result<DeviceSignedVector> {
        DeviceSignedVector::difference(
            self.positive.apply(device, vector)?,
            self.negative.apply(device, vector)?,
        )
    }
}

impl DeviceSignedVector {
    pub fn add(&self, device: &Device, right: &Self) -> Result<Self> {
        Self::difference(
            self.positive.add(device, &right.positive)?,
            self.negative.add(device, &right.negative)?,
        )
    }

    pub fn absolute(&self, device: &Device) -> Result<DeviceVector> {
        let output = DeviceVector::empty(device, self.positive.length);
        AbsoluteValue::launch(
            device,
            count(device, self.positive.length),
            CubeDim::new_1d(device.workgroup),
            AbsoluteValueInputLaunch {
                positive_input: argument(&self.positive.data),
                negative_input: argument(&self.negative.data),
                output: argument(&output.data),
                length: self.positive.length as u32,
            },
        );
        Ok(output)
    }
}

pub struct LaborDirection<'a> {
    pub goods: &'a DeviceSignedMatrix,
    pub direct_labor: &'a DeviceSignedMatrix,
}

pub struct ReproductionDirection<'a> {
    pub labor: &'a DeviceMatrix,
    pub reproduction: &'a DeviceMatrix,
    pub labor_change: &'a DeviceSignedMatrix,
    pub reproduction_change: &'a DeviceSignedMatrix,
    pub training_change: &'a DeviceSignedMatrix,
}

pub struct FiniteLaborChange<'a> {
    pub old_goods: &'a DeviceMatrix,
    pub new_goods: &'a DeviceMatrix,
    pub old_direct_labor: &'a DeviceMatrix,
    pub new_direct_labor: &'a DeviceMatrix,
    pub old_labor: &'a DeviceMatrix,
    pub new_inverse: &'a DeviceMatrix,
}

#[derive(Clone, Copy, Debug)]
pub struct CoefficientDirection {
    pub input: usize,
    pub output: usize,
}

pub struct Sensitivity;

impl Sensitivity {
    pub fn inverse(
        device: &Device,
        inverse: &DeviceMatrix,
        goods_change: &DeviceSignedMatrix,
    ) -> Result<DeviceSignedMatrix> {
        match inverse.rows == inverse.columns {
            true => goods_change
                .left_product(device, inverse)?
                .right_product(device, inverse),
            false => Err(anyhow!("inverse derivative requires a square inverse")),
        }
    }

    pub fn labor(
        device: &Device,
        labor: &DeviceMatrix,
        inverse: &DeviceMatrix,
        direction: LaborDirection<'_>,
    ) -> Result<DeviceSignedMatrix> {
        match inverse.rows == inverse.columns {
            true => direction
                .direct_labor
                .add(device, &direction.goods.left_product(device, labor)?)?
                .right_product(device, inverse),
            false => Err(anyhow!("labor derivative requires a square inverse")),
        }
    }

    pub fn finite_labor_change(
        device: &Device,
        change: FiniteLaborChange<'_>,
    ) -> Result<DeviceSignedMatrix> {
        let alias = |matrix: &DeviceMatrix| DeviceMatrix {
            data: matrix.data.clone(),
            rows: matrix.rows,
            columns: matrix.columns,
            scale: matrix.scale,
        };
        let goods =
            DeviceSignedMatrix::difference(alias(change.new_goods), alias(change.old_goods))?;
        let direct_labor = DeviceSignedMatrix::difference(
            alias(change.new_direct_labor),
            alias(change.old_direct_labor),
        )?;
        Self::labor(
            device,
            change.old_labor,
            change.new_inverse,
            LaborDirection {
                goods: &goods,
                direct_labor: &direct_labor,
            },
        )
    }

    pub fn coefficient(
        device: &Device,
        labor: &DeviceMatrix,
        inverse: &DeviceMatrix,
        direction: CoefficientDirection,
    ) -> Result<DeviceMatrix> {
        match inverse.rows == inverse.columns {
            true => {
                let matrix = SparseMatrix::new(
                    inverse.rows,
                    inverse.columns,
                    vec![Coefficient {
                        row: direction.input,
                        column: direction.output,
                        numerator: 1,
                        denominator: 1,
                    }],
                )?;
                labor
                    .product(device, &DeviceMatrix::new(device, &matrix, labor.scale)?)?
                    .product(device, inverse)
            }
            false => Err(anyhow!("coefficient sensitivity requires a square inverse")),
        }
    }

    pub fn reproduction(
        device: &Device,
        direction: ReproductionDirection<'_>,
    ) -> Result<DeviceSignedMatrix> {
        match direction.reproduction.columns == direction.labor.rows {
            true => direction
                .labor_change
                .right_product(device, direction.reproduction)?
                .add(
                    device,
                    &direction
                        .reproduction_change
                        .left_product(device, direction.labor)?,
                )?
                .add(device, direction.training_change),
            false => Err(anyhow!("reproduction derivative requires a square result")),
        }
    }

    pub fn conversion(
        device: &Device,
        transpose_inverse: &DeviceMatrix,
        beta: &DeviceVector,
        reproduction_change: &DeviceSignedMatrix,
    ) -> Result<DeviceSignedVector> {
        match transpose_inverse.rows == transpose_inverse.columns
            && beta.length == transpose_inverse.rows
        {
            true => reproduction_change
                .transpose(device)?
                .apply(device, beta)?
                .apply(device, transpose_inverse),
            false => Err(anyhow!(
                "conversion derivative requires matching beta and square inverse"
            )),
        }
    }

    pub fn normalized(
        device: &Device,
        beta: &DeviceVector,
        beta_change: &DeviceSignedVector,
        reference: usize,
        scale: u32,
    ) -> Result<Diagnostic<DeviceSignedVector>> {
        match reference < beta.length && beta.length == beta_change.positive.length && scale > 0 {
            false => Err(anyhow!(
                "normalized derivative dimensions, reference, or scale disagree"
            )),
            true => {
                let positive = DeviceVector::empty(device, beta.length);
                let negative = DeviceVector::empty(device, beta.length);
                let control = device.tensor(&[0; CONTROL_WORDS]);
                NormalizedDerivative::launch(
                    device,
                    count(device, beta.length),
                    CubeDim::new_1d(device.workgroup),
                    NormalizedDerivativeInputLaunch {
                        beta: argument(&beta.data),
                        change_positive: argument(&beta_change.positive.data),
                        change_negative: argument(&beta_change.negative.data),
                        positive_output: argument(&positive.data),
                        negative_output: argument(&negative.data),
                        control: argument(&control),
                        length: beta.length as u32,
                        reference: reference as u32,
                        scale,
                    },
                );
                let status = faults(device, &control)?;
                Ok(match status[0] {
                    0 => Diagnostic::Available(DeviceSignedVector::difference(positive, negative)?),
                    _ => Diagnostic::UnresolvedAccuracy,
                })
            }
        }
    }

    pub fn scalar_values(
        device: &Device,
        labor: &DeviceMatrix,
        weights: &DeviceVector,
        labor_change: &DeviceSignedMatrix,
        weight_change: &DeviceSignedVector,
    ) -> Result<DeviceSignedVector> {
        labor_change.transpose(device)?.apply(device, weights)?.add(
            device,
            &weight_change.apply(device, &labor.transpose(device)?)?,
        )
    }

    pub fn residual_error(
        device: &Device,
        inverse: &DeviceMatrix,
        residual: &DeviceSignedVector,
    ) -> Result<DeviceVector> {
        match inverse.rows == inverse.columns {
            true => inverse.apply(device, &residual.absolute(device)?),
            false => Err(anyhow!("residual bound requires a square inverse")),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NeumannTerms(u32);

impl NeumannTerms {
    pub fn new(terms: u32) -> Result<Self> {
        match terms <= 4096 {
            true => Ok(Self(terms)),
            false => Err(anyhow!("Neumann truncation exceeds 4096 retained terms")),
        }
    }

    pub fn count(self) -> u32 {
        self.0
    }
}

pub struct LaborTruncation {
    partial: DeviceMatrix,
    enclosure: DeviceMatrix,
    weighted_tail: DeviceVector,
    terms: NeumannTerms,
}

#[derive(Clone, Debug)]
pub struct TruncationReport {
    pub rows: usize,
    pub columns: usize,
    pub scale: u32,
    pub partial: Vec<Interval>,
    pub enclosure: Vec<Interval>,
    pub weighted_tail: u64,
    pub terms: u32,
    pub profile: &'static str,
}

pub struct SignedLaborTruncation {
    partial: DeviceSignedMatrix,
    enclosure: DeviceSignedMatrix,
    weighted_tail: DeviceVector,
    terms: NeumannTerms,
}

#[derive(Clone, Debug)]
pub struct SignedTruncationReport {
    pub rows: usize,
    pub columns: usize,
    pub scale: u32,
    pub partial: Vec<SignedInterval>,
    pub enclosure: Vec<SignedInterval>,
    pub weighted_tail: u64,
    pub terms: u32,
    pub profile: &'static str,
}

impl SignedLaborTruncation {
    pub fn new(
        device: &Device,
        goods: &SparseMatrix,
        labor: &DeviceSignedMatrix,
        contraction: &Contraction,
        terms: NeumannTerms,
    ) -> Result<Diagnostic<Self>> {
        let positive = LaborTruncation::new(device, goods, &labor.positive, contraction, terms)?;
        let negative = LaborTruncation::new(device, goods, &labor.negative, contraction, terms)?;
        Ok(match (positive, negative) {
            (Diagnostic::Available(positive), Diagnostic::Available(negative)) => {
                Diagnostic::Available(Self {
                    partial: DeviceSignedMatrix::difference(positive.partial, negative.partial)?,
                    enclosure: DeviceSignedMatrix::difference(
                        positive.enclosure,
                        negative.enclosure,
                    )?,
                    weighted_tail: positive
                        .weighted_tail
                        .add(device, &negative.weighted_tail)?,
                    terms,
                })
            }
            (Diagnostic::Inadmissible, _) | (_, Diagnostic::Inadmissible) => {
                Diagnostic::Inadmissible
            }
            _ => Diagnostic::UnresolvedAccuracy,
        })
    }

    pub fn partial(&self) -> &DeviceSignedMatrix {
        &self.partial
    }

    pub fn enclosure(&self) -> &DeviceSignedMatrix {
        &self.enclosure
    }

    pub fn read(&self, device: &Device) -> Result<SignedTruncationReport> {
        Ok(SignedTruncationReport {
            rows: self.partial.rows(),
            columns: self.partial.columns(),
            scale: self.partial.scale(),
            partial: self.partial.read(device)?,
            enclosure: self.enclosure.read(device)?,
            weighted_tail: self.weighted_tail.read(device)?[0].upper,
            terms: self.terms.0,
            profile: ANALYTICAL_PROFILE,
        })
    }
}

impl LaborTruncation {
    pub fn new(
        device: &Device,
        goods: &SparseMatrix,
        labor: &DeviceMatrix,
        contraction: &Contraction,
        terms: NeumannTerms,
    ) -> Result<Diagnostic<Self>> {
        let admitted = goods.rows == goods.columns
            && labor.columns == goods.rows
            && contraction.weights.len() == goods.rows;
        match admitted {
            false => Err(anyhow!("labor truncation dimensions or witness disagree")),
            true => {
                let sparse = DeviceOperator::new(device, goods);
                let weights = device.tensor(&padded(contraction.weights.clone()));
                let control = device.tensor(&[0; CONTROL_WORDS]);
                sparse.certificate(device, contraction, &control)?;
                let status = faults(device, &control)?;
                match status[0] {
                    kernels::AVAILABLE => {
                        let goods = DeviceMatrix::new(device, goods, labor.scale)?;
                        let mut partial = DeviceMatrix::new(
                            device,
                            &SparseMatrix::new(labor.rows, labor.columns, Vec::new())?,
                            labor.scale,
                        )?;
                        let mut term = DeviceMatrix {
                            data: labor.data.clone(),
                            rows: labor.rows,
                            columns: labor.columns,
                            scale: labor.scale,
                        };
                        for retained in 0..terms.0 {
                            partial = partial.add(device, &term)?;
                            if retained + 1 < terms.0 {
                                term = term.product(device, &goods)?;
                            }
                        }
                        let weighted = tensor::empty(device, [2, labor.rows, labor.columns]);
                        let weighted_tail = DeviceVector::empty(device, 1);
                        WeightedValues::launch(
                            device,
                            count(device, labor.rows * labor.columns),
                            CubeDim::new_1d(device.workgroup),
                            WeightedValuesInputLaunch {
                                labor: argument(&labor.data),
                                weights: argument(&weights),
                                output: argument(&weighted),
                                rows: labor.rows as u32,
                                columns: labor.columns as u32,
                            },
                        );
                        let weighted = TensorHandle::new_contiguous(
                            [2, labor.rows, labor.columns],
                            weighted.handle,
                            f32::elem_type_native(),
                        );
                        let norms = reduction::reduce(
                            device,
                            &weighted,
                            2,
                            ReduceOperationConfig::Sum,
                            f32::elem_type_native(),
                        )?;
                        let norms = TensorHandle::new_contiguous(
                            [labor.rows * 2],
                            norms.handle,
                            f32::elem_type_native(),
                        );
                        let norm = reduction::reduce(
                            device,
                            &norms,
                            0,
                            ReduceOperationConfig::Max,
                            f32::elem_type_native(),
                        )?;
                        TailBound::launch(
                            device,
                            CubeCount::Static(1, 1, 1),
                            CubeDim::new_1d(1),
                            TailBoundInputLaunch {
                                norms: norm.into_arg().into_buffer_arg(),
                                output: argument(&weighted_tail.data),
                                control: argument(&control),
                                numerator: contraction.numerator,
                                denominator: contraction.denominator,
                                terms: terms.0,
                            },
                        );
                        let enclosure =
                            DeviceMatrix::empty(device, labor.rows, labor.columns, labor.scale)?;
                        TailEnclosure::launch(
                            device,
                            count(device, labor.rows * labor.columns),
                            CubeDim::new_1d(device.workgroup),
                            TailEnclosureInputLaunch {
                                partial: argument(&partial.data),
                                tail: argument(&weighted_tail.data),
                                weights: argument(&weights),
                                output: argument(&enclosure.data),
                                control: argument(&control),
                                rows: labor.rows as u32,
                                columns: labor.columns as u32,
                            },
                        );
                        faults(device, &control)?;
                        Ok(Diagnostic::Available(Self {
                            partial,
                            enclosure,
                            weighted_tail,
                            terms,
                        }))
                    }
                    kernels::INADMISSIBLE => Ok(Diagnostic::Inadmissible),
                    _ => Ok(Diagnostic::UnresolvedAccuracy),
                }
            }
        }
    }

    pub fn partial(&self) -> &DeviceMatrix {
        &self.partial
    }

    pub fn enclosure(&self) -> &DeviceMatrix {
        &self.enclosure
    }

    pub fn read(&self, device: &Device) -> Result<TruncationReport> {
        Ok(TruncationReport {
            rows: self.partial.rows,
            columns: self.partial.columns,
            scale: self.partial.scale,
            partial: self.partial.read(device)?,
            enclosure: self.enclosure.read(device)?,
            weighted_tail: self.weighted_tail.read(device)?[0].upper,
            terms: self.terms.0,
            profile: ANALYTICAL_PROFILE,
        })
    }
}

mod derivative_kernels {
    use super::super::kernels::{Bounds, UNRESOLVED, divide_product, interval, scaled, sum, write};
    use super::super::kernels::{checked, positive};
    use cubecl::prelude::*;

    #[cube]
    pub fn absolute(
        positive_input: &[f32],
        negative_input: &[f32],
        output: &mut [f32],
        length: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < length as usize {
            let a = interval(positive_input, index);
            let b = interval(negative_input, index);
            write(
                output,
                index,
                Bounds {
                    lower: max(positive(a.lower, b.upper), positive(b.lower, a.upper)),
                    upper: max(positive(a.upper, b.lower), positive(b.upper, a.lower)),
                },
            );
        }
    }

    #[cube]
    fn quotient(value: Bounds, divisor: &Bounds, scale: u32, control: &[Atomic<u32>]) -> Bounds {
        Bounds {
            lower: divide_product(value.lower, scale as f32, divisor.upper, control).lower,
            upper: divide_product(value.upper, scale as f32, divisor.lower, control).upper,
        }
    }

    #[cube]
    fn cross_term(
        left: &Bounds,
        right: Bounds,
        divisor: &Bounds,
        control: &[Atomic<u32>],
    ) -> Bounds {
        Bounds {
            lower: divide_product(left.lower, right.lower, divisor.upper, control).lower,
            upper: divide_product(left.upper, right.upper, divisor.lower, control).upper,
        }
    }

    #[cube]
    pub fn normalized(
        beta: &[f32],
        change_positive: &[f32],
        change_negative: &[f32],
        positive_output: &mut [f32],
        negative_output: &mut [f32],
        control: &[Atomic<u32>],
        length: u32,
        reference: u32,
        scale: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < length as usize {
            let divisor = interval(beta, reference as usize);
            if divisor.lower == 0.0 {
                control[0].store(UNRESOLVED);
            }
            if index == reference as usize {
                write(
                    positive_output,
                    index,
                    Bounds {
                        lower: 0.0,
                        upper: 0.0,
                    },
                );
                write(
                    negative_output,
                    index,
                    Bounds {
                        lower: 0.0,
                        upper: 0.0,
                    },
                );
            }
            if index != reference as usize && !(divisor.lower == 0.0) {
                let alpha = quotient(interval(beta, index), &divisor, scale, control);
                let dp = quotient(interval(change_positive, index), &divisor, scale, control);
                let dn = quotient(interval(change_negative, index), &divisor, scale, control);
                let cp = cross_term(
                    &alpha,
                    interval(change_negative, reference as usize),
                    &divisor,
                    control,
                );
                let cn = cross_term(
                    &alpha,
                    interval(change_positive, reference as usize),
                    &divisor,
                    control,
                );
                let positive = Bounds {
                    lower: sum(dp.lower, cp.lower, control),
                    upper: sum(dp.upper, cp.upper, control),
                };
                let negative = Bounds {
                    lower: sum(dn.lower, cn.lower, control),
                    upper: sum(dn.upper, cn.upper, control),
                };
                write(positive_output, index, positive);
                write(negative_output, index, negative);
            }
        }
    }

    #[cube]
    pub fn weighted_values(
        labor: &[f32],
        weights: &[u32],
        output: &mut [f32],
        rows: u32,
        columns: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < rows as usize * columns as usize {
            let weighted = interval(labor, index).upper * weights[index % columns as usize] as f32;
            write(
                output,
                index,
                Bounds {
                    lower: weighted,
                    upper: weighted,
                },
            );
        }
    }

    #[cube]
    pub fn tail_bound(
        norms: &[f32],
        output: &mut [f32],
        control: &[Atomic<u32>],
        numerator: u32,
        denominator: u32,
        terms: u32,
    ) {
        if ABSOLUTE_POS == 0 {
            let mut bound = checked(max(norms[0], 0.0), control);
            for _term in 0..terms {
                bound = scaled(bound, numerator, denominator, control).upper;
            }
            bound = scaled(bound, denominator, denominator - numerator, control).upper;
            write(
                output,
                0usize,
                Bounds {
                    lower: 0.0,
                    upper: bound,
                },
            );
        }
    }

    #[cube]
    pub fn tail_enclosure(
        partial: &[f32],
        tail: &[f32],
        weights: &[u32],
        output: &mut [f32],
        control: &[Atomic<u32>],
        rows: u32,
        columns: u32,
    ) {
        let index = ABSOLUTE_POS;
        if index < rows as usize * columns as usize {
            let value = interval(partial, index);
            let bound = scaled(
                interval(tail, 0usize).upper,
                1u32,
                weights[index % columns as usize],
                control,
            )
            .upper;
            write(
                output,
                index,
                Bounds {
                    lower: value.lower,
                    upper: sum(value.upper, bound, control),
                },
            );
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct AbsoluteValueInput {
    pub(super) positive_input: Box<[f32]>,
    pub(super) negative_input: Box<[f32]>,
    pub(super) output: Box<[f32]>,
    pub(super) length: u32,
}

pub(super) struct AbsoluteValue;

pub(super) struct BoundAbsoluteValue;

impl System for AbsoluteValue {
    type Input = AbsoluteValueInput;
    type Passes = gpu_passes![BoundAbsoluteValue];
}

#[cube]
impl GpuPass<AbsoluteValueInput> for BoundAbsoluteValue {
    type Output = ();

    fn execute(mut input: AbsoluteValueInput) {
        derivative_kernels::absolute(
            &input.positive_input,
            &input.negative_input,
            &mut input.output,
            input.length,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct NormalizedDerivativeInput {
    pub(super) beta: Box<[f32]>,
    pub(super) change_positive: Box<[f32]>,
    pub(super) change_negative: Box<[f32]>,
    pub(super) positive_output: Box<[f32]>,
    pub(super) negative_output: Box<[f32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) length: u32,
    pub(super) reference: u32,
    pub(super) scale: u32,
}

pub(super) struct NormalizedDerivative;

pub(super) struct NormalizeDerivative;

impl System for NormalizedDerivative {
    type Input = NormalizedDerivativeInput;
    type Passes = gpu_passes![NormalizeDerivative];
}

#[cube]
impl GpuPass<NormalizedDerivativeInput> for NormalizeDerivative {
    type Output = ();

    fn execute(mut input: NormalizedDerivativeInput) {
        derivative_kernels::normalized(
            &input.beta,
            &input.change_positive,
            &input.change_negative,
            &mut input.positive_output,
            &mut input.negative_output,
            &input.control,
            input.length,
            input.reference,
            input.scale,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct WeightedValuesInput {
    pub(super) labor: Box<[f32]>,
    pub(super) weights: Box<[u32]>,
    pub(super) output: Box<[f32]>,
    pub(super) rows: u32,
    pub(super) columns: u32,
}

pub(super) struct WeightedValues;

pub(super) struct WeightValues;

impl System for WeightedValues {
    type Input = WeightedValuesInput;
    type Passes = gpu_passes![WeightValues];
}

#[cube]
impl GpuPass<WeightedValuesInput> for WeightValues {
    type Output = ();

    fn execute(mut input: WeightedValuesInput) {
        derivative_kernels::weighted_values(
            &input.labor,
            &input.weights,
            &mut input.output,
            input.rows,
            input.columns,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct TailBoundInput {
    pub(super) norms: Box<[f32]>,
    pub(super) output: Box<[f32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) numerator: u32,
    pub(super) denominator: u32,
    pub(super) terms: u32,
}

pub(super) struct TailBound;

pub(super) struct BoundTail;

impl System for TailBound {
    type Input = TailBoundInput;
    type Passes = gpu_passes![BoundTail];
}

#[cube]
impl GpuPass<TailBoundInput> for BoundTail {
    type Output = ();

    fn execute(mut input: TailBoundInput) {
        derivative_kernels::tail_bound(
            &input.norms,
            &mut input.output,
            &input.control,
            input.numerator,
            input.denominator,
            input.terms,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct TailEnclosureInput {
    pub(super) partial: Box<[f32]>,
    pub(super) tail: Box<[f32]>,
    pub(super) weights: Box<[u32]>,
    pub(super) output: Box<[f32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) rows: u32,
    pub(super) columns: u32,
}

pub(super) struct TailEnclosure;

pub(super) struct EncloseTail;

impl System for TailEnclosure {
    type Input = TailEnclosureInput;
    type Passes = gpu_passes![EncloseTail];
}

#[cube]
impl GpuPass<TailEnclosureInput> for EncloseTail {
    type Output = ();

    fn execute(mut input: TailEnclosureInput) {
        derivative_kernels::tail_enclosure(
            &input.partial,
            &input.tail,
            &input.weights,
            &mut input.output,
            &input.control,
            input.rows,
            input.columns,
        );
    }
}

use super::kernels;
use crate::simulation::{
    Device,
    pass::GpuPass,
    system::{System, gpu_passes},
};
use cubecl::prelude::*;
use std::marker::PhantomData;

#[derive(CubeType, CubeLaunch)]
pub(super) struct RescaleInput<N: Size> {
    pub(super) values: Box<[Vector<f32, N>]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) scale: f32,
    pub(super) length: u32,
}

pub(super) struct VectorRescaling<N: Size = Const<1>>(PhantomData<N>);
pub(super) struct RescaleVector;

impl VectorRescaling {
    pub(super) fn launch_vectorized(
        device: &Device,
        count: CubeCount,
        dimensions: CubeDim,
        input: RescaleInputLaunch<Const<1>>,
        width: usize,
    ) {
        let launch = match width {
            4 => Self::launch_sized::<Const<4>>,
            3 => Self::launch_sized::<Const<3>>,
            2 => Self::launch_sized::<Const<2>>,
            _ => Self::launch_sized::<Const<1>>,
        };
        launch(device, count, dimensions, input);
    }

    fn launch_sized<N: Size>(
        device: &Device,
        count: CubeCount,
        dimensions: CubeDim,
        input: RescaleInputLaunch<Const<1>>,
    ) {
        VectorRescaling::<N>::launch(
            device,
            count,
            dimensions,
            RescaleInputLaunch {
                values: input.values,
                control: input.control,
                scale: input.scale,
                length: input.length,
            },
        );
    }
}

impl<N: Size> System for VectorRescaling<N> {
    type Input = RescaleInput<N>;
    type Passes = gpu_passes![RescaleVector];
}

#[cube]
impl<N: Size> GpuPass<RescaleInput<N>> for RescaleVector {
    type Output = ();

    fn execute(mut input: RescaleInput<N>) {
        kernels::rescale::<N>(&mut input.values, &input.control, input.scale, input.length);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct AdditionInput<N: Size> {
    pub(super) left: Box<[Vector<f32, N>]>,
    pub(super) right: Box<[Vector<f32, N>]>,
    pub(super) output: Box<[Vector<f32, N>]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) length: u32,
}

pub(super) struct VectorAddition<N: Size = Const<1>>(PhantomData<N>);
pub(super) struct AddVectors;

impl VectorAddition {
    pub(super) fn launch_vectorized(
        device: &Device,
        count: CubeCount,
        dimensions: CubeDim,
        input: AdditionInputLaunch<Const<1>>,
        width: usize,
    ) {
        let launch = match width {
            4 => Self::launch_sized::<Const<4>>,
            3 => Self::launch_sized::<Const<3>>,
            2 => Self::launch_sized::<Const<2>>,
            _ => Self::launch_sized::<Const<1>>,
        };
        launch(device, count, dimensions, input);
    }

    fn launch_sized<N: Size>(
        device: &Device,
        count: CubeCount,
        dimensions: CubeDim,
        input: AdditionInputLaunch<Const<1>>,
    ) {
        VectorAddition::<N>::launch(
            device,
            count,
            dimensions,
            AdditionInputLaunch {
                left: input.left,
                right: input.right,
                output: input.output,
                control: input.control,
                length: input.length,
            },
        );
    }
}

impl<N: Size> System for VectorAddition<N> {
    type Input = AdditionInput<N>;
    type Passes = gpu_passes![AddVectors];
}

#[cube]
impl<N: Size> GpuPass<AdditionInput<N>> for AddVectors {
    type Output = ();

    fn execute(mut input: AdditionInput<N>) {
        kernels::vector_add::<N>(
            &input.left,
            &input.right,
            &mut input.output,
            &input.control,
            input.length,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct VectorNormalizationInput {
    pub(super) input: Box<[f32]>,
    pub(super) output: Box<[f32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) length: u32,
    pub(super) reference: u32,
    pub(super) scale: u32,
}

pub(super) struct VectorNormalization;
pub(super) struct NormalizeVector;

impl System for VectorNormalization {
    type Input = VectorNormalizationInput;
    type Passes = gpu_passes![NormalizeVector];
}

#[cube]
impl GpuPass<VectorNormalizationInput> for NormalizeVector {
    type Output = ();

    fn execute(mut input: VectorNormalizationInput) {
        kernels::normalize(
            &input.input,
            &mut input.output,
            &input.control,
            input.length,
            input.reference,
            input.scale,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct CertificateRowsInput {
    pub(super) product: Box<[f32]>,
    pub(super) weights: Box<[f32]>,
    pub(super) output: Box<[u32]>,
    pub(super) rows: u32,
    pub(super) numerator: u32,
    pub(super) denominator: u32,
}

pub(super) struct CertificateRows;
pub(super) struct CheckCertificateRows;

impl System for CertificateRows {
    type Input = CertificateRowsInput;
    type Passes = gpu_passes![CheckCertificateRows];
}

#[cube]
impl GpuPass<CertificateRowsInput> for CheckCertificateRows {
    type Output = ();

    fn execute(mut input: CertificateRowsInput) {
        kernels::certificate_rows(
            &input.product,
            &input.weights,
            &mut input.output,
            input.rows,
            input.numerator,
            input.denominator,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct CertificateStatusInput {
    pub(super) flags: Box<[u32]>,
    pub(super) control: Box<[Atomic<u32>]>,
}

pub(super) struct CertificateStatus;
pub(super) struct ResolveCertificate;

impl System for CertificateStatus {
    type Input = CertificateStatusInput;
    type Passes = gpu_passes![ResolveCertificate];
}

#[cube]
impl GpuPass<CertificateStatusInput> for ResolveCertificate {
    type Output = ();

    fn execute(input: CertificateStatusInput) {
        kernels::certificate_status(&input.flags, &input.control);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct SolutionResidualInput {
    pub(super) product: Box<[f32]>,
    pub(super) estimate: Box<[f32]>,
    pub(super) weights: Box<[u32]>,
    pub(super) output: Box<[f32]>,
    pub(super) rows: u32,
}

pub(super) struct SolutionResidual;
pub(super) struct MeasureResidual;

impl System for SolutionResidual {
    type Input = SolutionResidualInput;
    type Passes = gpu_passes![MeasureResidual];
}

#[cube]
impl GpuPass<SolutionResidualInput> for MeasureResidual {
    type Output = ();

    fn execute(mut input: SolutionResidualInput) {
        kernels::residual(
            &input.product,
            &input.estimate,
            &input.weights,
            &mut input.output,
            input.rows,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct SolutionErrorInput {
    pub(super) norm: Box<[f32]>,
    pub(super) error: Box<[f32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) numerator: u32,
    pub(super) denominator: u32,
    pub(super) tolerance: f32,
}

pub(super) struct SolutionError;
pub(super) struct BoundError;

impl System for SolutionError {
    type Input = SolutionErrorInput;
    type Passes = gpu_passes![BoundError];
}

#[cube]
impl GpuPass<SolutionErrorInput> for BoundError {
    type Output = ();

    fn execute(mut input: SolutionErrorInput) {
        kernels::error_bound(
            &input.norm,
            &mut input.error,
            &input.control,
            input.numerator,
            input.denominator,
            input.tolerance,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct SolutionBoundsInput {
    pub(super) estimate: Box<[f32]>,
    pub(super) weights: Box<[u32]>,
    pub(super) error: Box<[f32]>,
    pub(super) output: Box<[f32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) rows: u32,
}

pub(super) struct SolutionBounds;
pub(super) struct EncloseSolution;

impl System for SolutionBounds {
    type Input = SolutionBoundsInput;
    type Passes = gpu_passes![EncloseSolution];
}

#[cube]
impl GpuPass<SolutionBoundsInput> for EncloseSolution {
    type Output = ();

    fn execute(mut input: SolutionBoundsInput) {
        kernels::solution_bounds(
            &input.estimate,
            &input.weights,
            &input.error,
            &mut input.output,
            &input.control,
            input.rows,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct ValueComparisonInput {
    pub(super) requirement: Box<[f32]>,
    pub(super) output: Box<[u32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) living_low: u32,
    pub(super) living_high: u32,
    pub(super) scale: u32,
}

pub(super) struct ValueComparison;
pub(super) struct CompareValues;

impl System for ValueComparison {
    type Input = ValueComparisonInput;
    type Passes = gpu_passes![CompareValues];
}

#[cube]
impl GpuPass<ValueComparisonInput> for CompareValues {
    type Output = ();

    fn execute(mut input: ValueComparisonInput) {
        kernels::comparator(
            &input.requirement,
            &mut input.output,
            &input.control,
            input.living_low,
            input.living_high,
            input.scale,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct MatrixUnitsInput {
    pub(super) input: Box<[f32]>,
    pub(super) row_factors: Box<[u32]>,
    pub(super) column_factors: Box<[u32]>,
    pub(super) output: Box<[f32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) rows: u32,
    pub(super) columns: u32,
}

pub(super) struct MatrixUnits;
pub(super) struct ScaleMatrixUnits;

impl System for MatrixUnits {
    type Input = MatrixUnitsInput;
    type Passes = gpu_passes![ScaleMatrixUnits];
}

#[cube]
impl GpuPass<MatrixUnitsInput> for ScaleMatrixUnits {
    type Output = ();

    fn execute(mut input: MatrixUnitsInput) {
        kernels::dense_units(
            &input.input,
            &input.row_factors,
            &input.column_factors,
            &mut input.output,
            &input.control,
            input.rows,
            input.columns,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct MatrixColumnInput {
    pub(super) input: Box<[f32]>,
    pub(super) output: Box<[f32]>,
    pub(super) rows: u32,
    pub(super) columns: u32,
    pub(super) column: u32,
}

pub(super) struct MatrixColumn;
pub(super) struct StoreMatrixColumn;

impl System for MatrixColumn {
    type Input = MatrixColumnInput;
    type Passes = gpu_passes![StoreMatrixColumn];
}

#[cube]
impl GpuPass<MatrixColumnInput> for StoreMatrixColumn {
    type Output = ();

    fn execute(mut input: MatrixColumnInput) {
        kernels::dense_column(
            &input.input,
            &mut input.output,
            input.rows,
            input.columns,
            input.column,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct SignedVectorReadInput {
    pub(super) positive: Box<[f32]>,
    pub(super) negative: Box<[f32]>,
    pub(super) output: Box<[u32]>,
    pub(super) control: Box<[Atomic<u32>]>,
    pub(super) length: u32,
}

pub(super) struct SignedVectorRead;
pub(super) struct EncodeSignedVector;

impl System for SignedVectorRead {
    type Input = SignedVectorReadInput;
    type Passes = gpu_passes![EncodeSignedVector];
}

#[cube]
impl GpuPass<SignedVectorReadInput> for EncodeSignedVector {
    type Output = ();

    fn execute(mut input: SignedVectorReadInput) {
        kernels::signed_vector(
            &input.positive,
            &input.negative,
            &mut input.output,
            &input.control,
            input.length,
        );
    }
}

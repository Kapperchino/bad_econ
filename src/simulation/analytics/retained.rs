use super::*;

impl DeviceVector {
    pub fn scale_units(&self, device: &Device, factors: &[UnitFactor]) -> Result<Self> {
        let matrix = DeviceMatrix {
            data: self.data.clone(),
            rows: self.length,
            columns: 1,
            scale: 1,
        };
        Ok(Self {
            data: matrix
                .scale_units(device, factors, &[UnitFactor::new(1, 1)?])?
                .data,
            length: self.length,
        })
    }

    pub fn reciprocal_reference(
        &self,
        device: &Device,
        reference: usize,
        scale: u32,
    ) -> Result<Diagnostic<Self>> {
        match reference < self.length && scale > 0 {
            false => Err(anyhow!(
                "reciprocal reference requires an existing index and positive scale"
            )),
            true => {
                let output = Self::empty(device, 1);
                let control = device.tensor(&[0; CONTROL_WORDS]);
                retained_kernels::ReciprocalReference::launch(
                    device,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    retained_kernels::ReciprocalInputLaunch {
                        values: argument(&self.data),
                        output: argument(&output.data),
                        control: argument(&control),
                        reference: reference as u32,
                        scale,
                    },
                );
                Ok(match faults(device, &control)?[0] {
                    0 => Diagnostic::Available(output),
                    1 => Diagnostic::Inapplicable,
                    _ => Diagnostic::UnresolvedAccuracy,
                })
            }
        }
    }
}

impl DeviceSignedVector {
    pub fn scale_units(&self, device: &Device, factors: &[UnitFactor]) -> Result<Self> {
        Self::difference(
            self.positive.scale_units(device, factors)?,
            self.negative.scale_units(device, factors)?,
        )
    }
}

pub struct InventoryRegime<'a> {
    pub context: ContextVersion,
    pub goods: &'a DeviceOperator,
    pub direct_labor: &'a DeviceOperator,
    pub labor_requirements: &'a DeviceMatrix,
}

pub struct InventoryStage<'a> {
    pub context: ContextVersion,
    pub closing: &'a DeviceVector,
    pub production: &'a DeviceVector,
    pub final_demand: &'a DeviceVector,
    pub research_inputs: &'a DeviceVector,
    pub direct_research_hours: &'a DeviceVector,
}

pub struct InventoryHorizon {
    pub context: ContextVersion,
    pub stages: usize,
    pub physical_residuals: Vec<DeviceSignedVector>,
    pub actual_hours: DeviceVector,
    pub final_and_project_hours: DeviceVector,
    pub accumulated_stock_adjustment: DeviceSignedVector,
    pub terminal_stock_adjustment: DeviceSignedVector,
    pub reconciled_hours: DeviceSignedVector,
    pub reconciliation_residual: DeviceSignedVector,
}

impl InventoryHorizon {
    pub fn new(
        device: &Device,
        regime: InventoryRegime<'_>,
        opening: &DeviceVector,
        stages: &[InventoryStage<'_>],
    ) -> Result<Self> {
        let n = regime.goods.rows;
        let m = regime.direct_labor.rows;
        let admitted = regime.goods.columns == n
            && regime.direct_labor.columns == n
            && regime.labor_requirements.rows == m
            && regime.labor_requirements.columns == n
            && opening.length == n
            && stages.len() <= 4096
            && stages.iter().all(|stage| stage.context == regime.context);
        match admitted {
            false => Err(anyhow!(
                "inventory horizon requires one context, matching dimensions, and at most 4096 stages"
            )),
            true => {
                let zero = || DeviceVector::from_ticks(device, &vec![0; m]);
                let mut current = opening;
                let mut actual_hours = zero()?;
                let mut final_and_project_hours = zero()?;
                let mut positive_adjustment = zero()?;
                let mut negative_adjustment = zero()?;
                let mut physical_residuals = Vec::with_capacity(stages.len());
                for stage in stages {
                    let account = InventoryAccount::new(
                        device,
                        regime.goods,
                        regime.direct_labor,
                        regime.labor_requirements,
                        InventoryFlows {
                            opening: current,
                            closing: stage.closing,
                            production: stage.production,
                            final_demand: stage.final_demand,
                            research_inputs: stage.research_inputs,
                            direct_research_hours: stage.direct_research_hours,
                        },
                    )?;
                    actual_hours = actual_hours.add(device, &account.actual_hours)?;
                    final_and_project_hours = final_and_project_hours.add(
                        device,
                        &regime
                            .labor_requirements
                            .apply(
                                device,
                                &stage.final_demand.add(device, stage.research_inputs)?,
                            )?
                            .add(device, stage.direct_research_hours)?,
                    )?;
                    positive_adjustment = positive_adjustment.add(
                        device,
                        &regime.labor_requirements.apply(device, stage.closing)?,
                    )?;
                    negative_adjustment = negative_adjustment
                        .add(device, &regime.labor_requirements.apply(device, current)?)?;
                    physical_residuals.push(account.physical_residual);
                    current = stage.closing;
                }
                let terminal = regime.labor_requirements.apply(device, current)?;
                let initial = regime.labor_requirements.apply(device, opening)?;
                let reconciled_hours = DeviceSignedVector::difference(
                    final_and_project_hours.add(device, &terminal)?,
                    DeviceVector {
                        data: initial.data.clone(),
                        length: initial.length,
                    },
                )?;
                let reconciliation_residual = DeviceSignedVector::difference(
                    actual_hours.add(device, &initial)?,
                    final_and_project_hours.add(device, &terminal)?,
                )?;
                Ok(Self {
                    context: regime.context,
                    stages: stages.len(),
                    physical_residuals,
                    actual_hours,
                    final_and_project_hours,
                    accumulated_stock_adjustment: DeviceSignedVector::difference(
                        positive_adjustment,
                        negative_adjustment,
                    )?,
                    terminal_stock_adjustment: DeviceSignedVector::difference(terminal, initial)?,
                    reconciled_hours,
                    reconciliation_residual,
                })
            }
        }
    }
}

pub struct BoundaryDemand<'a> {
    pub retained: &'a DeviceVector,
    pub outside: &'a DeviceVector,
    pub retained_labor: &'a DeviceMatrix,
    pub outside_direct_labor: &'a DeviceMatrix,
    pub outside_inverse: &'a DeviceMatrix,
    pub retained_from_outside: &'a DeviceMatrix,
}

pub struct BoundaryDemandAccount {
    pub outside_requirements: DeviceVector,
    pub effective_retained_demand: DeviceVector,
    pub outside_direct_hours: DeviceVector,
    pub total_hours: DeviceVector,
}

impl BoundaryDemandAccount {
    pub fn new(device: &Device, demand: BoundaryDemand<'_>) -> Result<Self> {
        let outside_requirements = demand.outside_inverse.apply(device, demand.outside)?;
        let effective_retained_demand = demand.retained.add(
            device,
            &demand
                .retained_from_outside
                .apply(device, &outside_requirements)?,
        )?;
        let outside_direct_hours = demand
            .outside_direct_labor
            .apply(device, &outside_requirements)?;
        let total_hours = demand
            .retained_labor
            .apply(device, &effective_retained_demand)?
            .add(device, &outside_direct_hours)?;
        Ok(Self {
            outside_requirements,
            effective_retained_demand,
            outside_direct_hours,
            total_hours,
        })
    }
}

pub struct AugmentedBlocks<'a> {
    pub goods_inverse: &'a DeviceMatrix,
    pub reproduction: &'a DeviceMatrix,
    pub direct_labor: &'a DeviceMatrix,
    pub reproduction_inverse: &'a DeviceMatrix,
}

pub struct AugmentedInverse {
    pub goods_from_goods: DeviceMatrix,
    pub goods_from_labor: DeviceMatrix,
    pub labor_from_goods: DeviceMatrix,
    pub labor_from_labor: DeviceMatrix,
}

pub struct AugmentedRequirements {
    pub goods: DeviceVector,
    pub labor: DeviceVector,
}

impl AugmentedInverse {
    pub fn new(device: &Device, blocks: AugmentedBlocks<'_>) -> Result<Self> {
        let admitted = blocks.goods_inverse.rows == blocks.goods_inverse.columns
            && blocks.reproduction_inverse.rows == blocks.reproduction_inverse.columns;
        match admitted {
            false => Err(anyhow!("augmented inverse requires square inverse blocks")),
            true => {
                let labor = blocks.direct_labor.product(device, blocks.goods_inverse)?;
                let goods_from_labor = blocks
                    .goods_inverse
                    .product(device, blocks.reproduction)?
                    .product(device, blocks.reproduction_inverse)?;
                Ok(Self {
                    goods_from_goods: blocks
                        .goods_inverse
                        .add(device, &goods_from_labor.product(device, &labor)?)?,
                    goods_from_labor,
                    labor_from_goods: blocks.reproduction_inverse.product(device, &labor)?,
                    labor_from_labor: DeviceMatrix {
                        data: blocks.reproduction_inverse.data.clone(),
                        rows: blocks.reproduction_inverse.rows,
                        columns: blocks.reproduction_inverse.columns,
                        scale: blocks.reproduction_inverse.scale,
                    },
                })
            }
        }
    }

    pub fn apply(
        &self,
        device: &Device,
        goods: &DeviceVector,
        labor: &DeviceVector,
    ) -> Result<AugmentedRequirements> {
        Ok(AugmentedRequirements {
            goods: self
                .goods_from_goods
                .apply(device, goods)?
                .add(device, &self.goods_from_labor.apply(device, labor)?)?,
            labor: self
                .labor_from_goods
                .apply(device, goods)?
                .add(device, &self.labor_from_labor.apply(device, labor)?)?,
        })
    }
}

mod retained_kernels {
    use super::super::kernels::{Bounds, divide_product, interval, write};
    use crate::simulation::{
        pass::GpuPass,
        system::{System, gpu_passes},
    };
    use cubecl::prelude::*;

    #[derive(CubeType, CubeLaunch)]
    pub(super) struct ReciprocalInput {
        pub(super) values: Box<[f32]>,
        pub(super) output: Box<[f32]>,
        pub(super) control: Box<[Atomic<u32>]>,
        pub(super) reference: u32,
        pub(super) scale: u32,
    }

    pub(super) struct ReciprocalReference;
    pub(super) struct InvertReference;

    impl System for ReciprocalReference {
        type Input = ReciprocalInput;
        type Passes = gpu_passes![InvertReference];
    }

    #[cube]
    impl GpuPass<ReciprocalInput> for InvertReference {
        type Output = ();

        fn execute(mut input: ReciprocalInput) {
            reciprocal_reference(
                &input.values,
                &mut input.output,
                &input.control,
                input.reference,
                input.scale,
            );
        }
    }

    #[cube]
    fn reciprocal_reference(
        input: &[f32],
        output: &mut [f32],
        control: &[Atomic<u32>],
        reference: u32,
        scale: u32,
    ) {
        if ABSOLUTE_POS == 0 {
            let divisor = interval(input, reference as usize);
            if divisor.upper == 0.0 {
                control[0].store(1);
            }
            if (divisor.lower == 0.0) && !(divisor.upper == 0.0) {
                control[0].store(2);
            }
            if !(divisor.lower == 0.0) {
                let units = scale as f32;
                write(
                    output,
                    0usize,
                    Bounds {
                        lower: divide_product(units, units, divisor.upper, control).lower,
                        upper: divide_product(units, units, divisor.lower, control).upper,
                    },
                );
            }
        }
    }
}

use anyhow::{Result, anyhow};
use bad_econ::simulation::Device;
use clap::Parser;
use cubecl::{features::Plane, ir::UIntKind, prelude::*, std::tensor::TensorHandle};
use cubek::reduce::{
    ReduceDtypes, ReduceStrategy,
    components::instructions::ReduceOperationConfig,
    launch::{RoutineStrategy, VectorizationStrategy},
    routines::{BlueprintStrategy, cube::CubeStrategy, plane::PlaneStrategy, unit::UnitStrategy},
};
use std::{num::NonZeroU32, path::PathBuf, time::Instant};

#[derive(Parser)]
struct Arguments {
    #[arg(long, default_value = "64")]
    repeats: NonZeroU32,
    #[arg(long, default_value = "3")]
    trials: NonZeroU32,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Unit,
    Plane,
    Cube,
    CubePlane,
}

impl Kernel {
    fn strategy(self) -> ReduceStrategy {
        let routine = match self {
            Self::Unit => RoutineStrategy::Unit(BlueprintStrategy::Inferred(UnitStrategy)),
            Self::Plane => RoutineStrategy::Plane(BlueprintStrategy::Inferred(PlaneStrategy {
                independent: false,
            })),
            Self::Cube => RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                use_planes: false,
            })),
            Self::CubePlane => RoutineStrategy::Cube(BlueprintStrategy::Inferred(CubeStrategy {
                use_planes: true,
            })),
        };
        ReduceStrategy {
            routine,
            vectorization: VectorizationStrategy {
                parallel_output_vectorization: true,
            },
            autotune_level: Default::default(),
        }
    }
}

struct ReductionCase {
    name: &'static str,
    shape: Vec<usize>,
    axis: usize,
    dtypes: ReduceDtypes,
    operation: ReduceOperationConfig,
}

impl ReductionCase {
    fn float(name: &'static str, shape: Vec<usize>, axis: usize) -> Self {
        Self {
            name,
            shape,
            axis,
            dtypes: ReduceDtypes {
                input: f32::elem_type_native(),
                output: f32::elem_type_native(),
                accumulation: f32::elem_type_native(),
            },
            operation: match axis {
                0 => ReduceOperationConfig::Max,
                _ => ReduceOperationConfig::Sum,
            },
        }
    }

    fn integer(name: &'static str, shape: Vec<usize>, dtype: ElemType) -> Self {
        Self {
            name,
            shape,
            axis: 0,
            dtypes: ReduceDtypes {
                input: u32::elem_type_native(),
                output: dtype,
                accumulation: dtype,
            },
            operation: match dtype {
                ElemType::UInt(UIntKind::U64) => ReduceOperationConfig::Sum,
                _ => ReduceOperationConfig::All,
            },
        }
    }

    fn input(&self, device: &Device) -> TensorHandle {
        let elements = self.shape.iter().product::<usize>();
        let bytes = match self.dtypes.input {
            ElemType::Float(_) => bytemuck::cast_slice(&vec![1.0f32; elements]).to_vec(),
            _ => bytemuck::cast_slice(&vec![1u32; elements]).to_vec(),
        };
        TensorHandle::new_contiguous(
            self.shape.clone(),
            device.client.create_from_slice(&bytes),
            self.dtypes.input,
        )
    }

    fn check(&self, bytes: &[u8]) -> Result<()> {
        let expected = match self.operation {
            ReduceOperationConfig::Sum => self.shape[self.axis],
            _ => 1,
        };
        let valid = match self.dtypes.output {
            ElemType::Float(_) => bytemuck::cast_slice::<u8, f32>(bytes)
                .iter()
                .all(|&value| value == expected as f32),
            ElemType::UInt(UIntKind::U64) => bytemuck::cast_slice::<u8, u64>(bytes)
                .iter()
                .all(|&value| value == expected as u64),
            _ => bytemuck::cast_slice::<u8, u32>(bytes)
                .iter()
                .all(|&value| value == expected as u32),
        };
        match valid {
            true => Ok(()),
            false => Err(anyhow!("{} reduction result is incorrect", self.name)),
        }
    }
}

fn main() -> Result<()> {
    let args = Arguments::parse();
    let device = Device::headless(64)?;
    let cases = [
        ReductionCase::integer("population", vec![1024, 8], u64::elem_type_native()),
        ReductionCase::integer("research-small", vec![257, 21, 2], u64::elem_type_native()),
        ReductionCase::integer(
            "research-large",
            vec![16384, 21, 2],
            u64::elem_type_native(),
        ),
        ReductionCase::integer("certificate-small", vec![16, 2], u32::elem_type_native()),
        ReductionCase::integer("certificate-medium", vec![128, 2], u32::elem_type_native()),
        ReductionCase::integer("certificate-large", vec![4097, 2], u32::elem_type_native()),
        ReductionCase::float("maximum-small", vec![16], 0),
        ReductionCase::float("maximum-medium", vec![128], 0),
        ReductionCase::float("maximum-large", vec![4097], 0),
        ReductionCase::float("maximum-million", vec![1_000_001], 0),
        ReductionCase::float("rows-small", vec![2, 16, 16], 2),
        ReductionCase::float("rows-medium", vec![2, 128, 128], 2),
        ReductionCase::float("rows-large", vec![2, 512, 512], 2),
        ReductionCase::float("rows-tail", vec![2, 257, 127], 2),
        ReductionCase::float("rows-thousand", vec![2, 1024, 1024], 2),
    ];
    let mut reports = Vec::new();
    for case in cases {
        let input = case.input(&device);
        let mut output_shape = case.shape.clone();
        output_shape[case.axis] = 1;
        let output = TensorHandle::new_contiguous(
            output_shape.clone(),
            device
                .client
                .empty(output_shape.iter().product::<usize>() * case.dtypes.output.size()),
            case.dtypes.output,
        );
        let plane_support = device
            .client
            .properties()
            .features
            .plane
            .contains(Plane::Ops)
            && case.dtypes.accumulation.size() == 4;
        let kernels = match plane_support {
            true => vec![Kernel::Unit, Kernel::Plane, Kernel::Cube, Kernel::CubePlane],
            false => vec![Kernel::Unit, Kernel::Cube],
        };
        for kernel in kernels {
            let launch = || {
                cubek::reduce::reduce(
                    &device.client,
                    input.clone().binding(),
                    output.clone().binding(),
                    case.axis,
                    kernel.strategy(),
                    case.operation,
                    case.dtypes,
                )
            };
            launch()?;
            case.check(&device.client.read_one(output.handle.clone())?)?;
            launch()?;
            device.sync()?;
            let mut milliseconds = Vec::new();
            for _ in 0..args.trials.get() {
                let start = Instant::now();
                for _ in 0..args.repeats.get() {
                    launch()?;
                }
                device.sync()?;
                milliseconds
                    .push(start.elapsed().as_secs_f64() * 1000.0 / f64::from(args.repeats.get()));
            }
            case.check(&device.client.read_one(output.handle.clone())?)?;
            eprintln!("{} {kernel:?}: {milliseconds:?} ms", case.name);
            reports.push(serde_json::json!({
                "case": case.name,
                "shape": case.shape,
                "axis": case.axis,
                "dtype": format!("{:?}", case.dtypes),
                "operation": format!("{:?}", case.operation),
                "device": device.name,
                "kernel": format!("{kernel:?}"),
                "repeats_per_trial": args.repeats,
                "milliseconds": milliseconds,
            }));
            std::fs::write(&args.output, serde_json::to_vec_pretty(&reports)?)?;
        }
    }
    Ok(())
}

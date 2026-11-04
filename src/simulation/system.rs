use super::{Device, pass::GpuPass};
use cubecl::prelude::*;
use std::marker::PhantomData;

pub struct PassList<First, Next> {
    first: PhantomData<First>,
    next: PhantomData<Next>,
}

pub struct End;

#[macro_export]
macro_rules! gpu_passes {
    () => {
        $crate::simulation::system::End
    };
    ($first:ty $(, $next:ty)* $(,)?) => {
        $crate::simulation::system::PassList<$first, $crate::gpu_passes![$($next),*]>
    };
}

pub use crate::gpu_passes;

#[cube]
impl<Input: CubeType> GpuPass<Input> for End {
    type Output = Input;

    fn execute(input: Input) -> Input {
        input
    }
}

#[cube]
impl<Input: CubeType, First: GpuPass<Input>, Next: GpuPass<First::Output>> GpuPass<Input>
    for PassList<First, Next>
{
    type Output = Next::Output;

    fn execute(input: Input) -> Self::Output {
        Next::execute(First::execute(input))
    }
}

pub trait System: Sized + Send + Sync + 'static {
    type Input: LaunchArg;
    type Passes: GpuPass<Self::Input, Output = ()>;

    fn launch(
        device: &Device,
        count: CubeCount,
        dimensions: CubeDim,
        input: <Self::Input as LaunchArg>::RuntimeArg,
    ) {
        launch_system::launch::<Self>(&device.client, count, dimensions, input);
    }
}

#[cube(launch)]
fn launch_system<S: System>(input: S::Input) {
    S::Passes::execute(input);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::parallel;
    use anyhow::Result;

    #[derive(CubeType, CubeLaunch)]
    struct Buffers {
        values: Box<[u32]>,
        output: Box<[u32]>,
        totals: Box<[u32]>,
    }

    #[derive(CubeType)]
    struct Counted {
        buffers: Buffers,
        present: u32,
    }

    #[derive(CubeType)]
    struct Scanned {
        buffers: Buffers,
        prefix: parallel::PrefixSum,
    }

    struct Count;
    struct PrefixSum;
    struct Store;
    struct CountingSystem;

    impl System for CountingSystem {
        type Input = Buffers;
        type Passes = gpu_passes![Count, PrefixSum, Store];
    }

    #[cube]
    impl GpuPass<Buffers> for Count {
        type Output = Counted;

        fn execute(input: Buffers) -> Counted {
            let mut present = 0u32;
            if ABSOLUTE_POS < input.values.len() {
                present = u32::cast_from(input.values[ABSOLUTE_POS] > 0);
            }
            Counted {
                buffers: input,
                present,
            }
        }
    }

    #[cube]
    impl GpuPass<Counted> for PrefixSum {
        type Output = Scanned;

        fn execute(input: Counted) -> Scanned {
            Scanned {
                buffers: input.buffers,
                prefix: parallel::prefix_sum(input.present),
            }
        }
    }

    #[cube]
    impl GpuPass<Scanned> for Store {
        type Output = ();

        fn execute(mut input: Scanned) {
            if ABSOLUTE_POS < input.buffers.output.len() {
                input.buffers.output[ABSOLUTE_POS] = input.prefix.inclusive;
            }
            if UNIT_POS == 0 {
                input.buffers.totals[CUBE_POS] = input.prefix.total;
            }
        }
    }

    #[test]
    fn system_launches_its_pass_list_on_gpu() -> Result<()> {
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            for count in [1, workgroup as usize, workgroup as usize * 2 + 1] {
                let values = (0..count)
                    .map(|index| (index % 3) as u32)
                    .collect::<Vec<_>>();
                let input = device.tensor(&values);
                let output = device.empty(count);
                let totals = device.empty(count.div_ceil(workgroup as usize));
                CountingSystem::launch(
                    &device,
                    device.count(count),
                    CubeDim::new_1d(workgroup),
                    BuffersLaunch {
                        values: input.into_buffer_arg(),
                        output: output.clone().into_buffer_arg(),
                        totals: totals.clone().into_buffer_arg(),
                    },
                );
                let expected = values
                    .chunks(workgroup as usize)
                    .flat_map(|block| {
                        block.iter().scan(0, |sum, value| {
                            *sum += u32::from(*value > 0);
                            Some(*sum)
                        })
                    })
                    .collect::<Vec<_>>();
                let expected_totals = values
                    .chunks(workgroup as usize)
                    .map(|block| block.iter().filter(|value| **value > 0).count() as u32)
                    .collect::<Vec<_>>();
                assert_eq!(device.read(&output)?, expected);
                assert_eq!(device.read(&totals)?, expected_totals);
            }
        }
        Ok(())
    }
}

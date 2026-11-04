use cubecl::prelude::*;

#[cube]
pub trait GpuPass<Input: CubeType>: Send + Sync + 'static {
    type Output: CubeType;

    fn execute(input: Input) -> Self::Output;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{Device, parallel};
    use anyhow::Result;

    struct Load;
    struct CountPresent;
    struct PrefixSum;

    #[cube]
    impl<'a> GpuPass<&'a [u32]> for Load {
        type Output = u32;

        fn execute(input: &'a [u32]) -> u32 {
            let mut value = 0u32;
            if ABSOLUTE_POS < input.len() {
                value = input[ABSOLUTE_POS];
            }
            value
        }
    }

    #[cube]
    impl GpuPass<u32> for CountPresent {
        type Output = u32;

        fn execute(input: u32) -> u32 {
            u32::cast_from(input > 0)
        }
    }

    #[cube]
    impl GpuPass<u32> for PrefixSum {
        type Output = parallel::PrefixSum;

        fn execute(input: u32) -> parallel::PrefixSum {
            parallel::prefix_sum(input)
        }
    }

    #[cube(launch)]
    fn fused<A: GpuPass<u32>, B: GpuPass<A::Output, Output = parallel::PrefixSum>>(
        input: &[u32],
        output: &mut [u32],
        totals: &mut [u32],
    ) {
        let id = ABSOLUTE_POS;
        let prefix = B::execute(A::execute(Load::execute(input)));
        if id < output.len() {
            output[id] = prefix.inclusive;
        }
        if UNIT_POS == 0 {
            totals[CUBE_POS] = prefix.total;
        }
    }

    #[test]
    fn typed_passes_compose_in_one_gpu_launch() -> Result<()> {
        for workgroup in [32, 64, 128, 256] {
            let device = Device::headless(workgroup)?;
            for count in [1, workgroup as usize, workgroup as usize * 2 + 1] {
                let values = (0..count)
                    .map(|index| (index % 3) as u32)
                    .collect::<Vec<_>>();
                let input = device.tensor(&values);
                let output = device.empty(count);
                let totals = device.empty(count.div_ceil(workgroup as usize));
                fused::launch::<CountPresent, PrefixSum>(
                    &device.client,
                    device.count(count),
                    CubeDim::new_1d(workgroup),
                    input.into_buffer_arg(),
                    output.clone().into_buffer_arg(),
                    totals.clone().into_buffer_arg(),
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

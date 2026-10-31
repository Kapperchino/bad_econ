use super::super::words::{divide, times32};
use cubecl::prelude::*;

pub const PROBABILITY_SCALE: u32 = 1_000_000;
pub const HAZARD_TABLE: [u32; 257] = [
    0, 60586, 117503, 170970, 221199, 268384, 312710, 354351, 393469, 430217, 464738, 497168,
    527633, 556252, 583137, 608394, 632120, 654409, 675347, 695017, 713495, 730853, 747160, 762479,
    776869, 790388, 803088, 815018, 826226, 836754, 846645, 855936, 864664, 872864, 880567, 887803,
    894600, 900986, 906985, 912620, 917915, 922888, 927560, 931949, 936072, 939945, 943583, 947001,
    950212, 953229, 956063, 958725, 961225, 963575, 965781, 967855, 969802, 971632, 973350, 974965,
    976482, 977907, 979245, 980503, 981684, 982794, 983836, 984815, 985735, 986599, 987411, 988174,
    988891, 989564, 990196, 990790, 991348, 991872, 992364, 992827, 993262, 993670, 994053, 994414,
    994752, 995070, 995369, 995649, 995913, 996160, 996393, 996611, 996817, 997010, 997191, 997361,
    997521, 997671, 997812, 997945, 998069, 998186, 998296, 998399, 998496, 998587, 998673, 998753,
    998829, 998900, 998966, 999029, 999088, 999143, 999195, 999244, 999289, 999332, 999373, 999411,
    999446, 999480, 999511, 999541, 999569, 999595, 999619, 999642, 999664, 999684, 999703, 999721,
    999738, 999754, 999769, 999783, 999796, 999808, 999820, 999831, 999841, 999851, 999860, 999868,
    999876, 999884, 999891, 999897, 999903, 999909, 999915, 999920, 999925, 999929, 999933, 999937,
    999941, 999945, 999948, 999951, 999954, 999957, 999959, 999962, 999964, 999966, 999968, 999970,
    999972, 999974, 999975, 999977, 999978, 999979, 999981, 999982, 999983, 999984, 999985, 999986,
    999986, 999987, 999988, 999989, 999989, 999990, 999991, 999991, 999992, 999992, 999993, 999993,
    999993, 999994, 999994, 999994, 999995, 999995, 999995, 999996, 999996, 999996, 999996, 999996,
    999997, 999997, 999997, 999997, 999997, 999997, 999998, 999998, 999998, 999998, 999998, 999998,
    999998, 999998, 999998, 999998, 999998, 999998, 999999, 999999, 999999, 999999, 999999, 999999,
    999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999,
    999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999,
    999999, 999999, 999999, 999999, 999999,
];

#[cube]
pub fn hazard(exposure: u32, table: &[u32]) -> u32 {
    let position = min(exposure, 16000) * 16;
    let index = min(position / 1000, 255);
    let remainder = position - index * 1000;
    let lower = table[index as usize];
    let upper = table[index as usize + 1];
    lower + (upper - lower) * remainder / 1000
}

#[cube]
pub fn sigmoid(score: i32, table: &[u32]) -> u32 {
    let magnitude = u32::cast_from(max(score, -score));
    let exponential = PROBABILITY_SCALE - hazard(magnitude, table);
    let denominator = PROBABILITY_SCALE + exponential;
    let numerator = select(score >= 0, PROBABILITY_SCALE, exponential);
    divide(times32(numerator, PROBABILITY_SCALE), denominator).extract(0usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{arithmetic::hex, domain::research::RESEARCH_TABLE_HASH, simulation::Device};
    use anyhow::Result;
    use sha2::{Digest, Sha256};

    #[cube(launch)]
    fn probe(table: &[u32], output: &mut [u32]) {
        let id = ABSOLUTE_POS;
        if id <= 16000 {
            output[id] = hazard(id as u32, table);
        }
        if id <= 9000 {
            output[16001 + id] = sigmoid(i32::cast_from(id) - 4500, table);
        }
    }

    fn check(device: Device) -> Result<()> {
        assert_eq!(
            hex(&Sha256::digest(bytemuck::cast_slice(&HAZARD_TABLE))),
            RESEARCH_TABLE_HASH
        );
        let table = device.tensor(&HAZARD_TABLE);
        let output = device.empty(25002);
        let argument = |tensor: &burn_wgpu::CubeTensor| unsafe {
            BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
        };
        probe::launch(
            &device.client,
            CubeCount::Static(16001u32.div_ceil(device.workgroup), 1, 1),
            CubeDim::new_1d(device.workgroup),
            argument(&table),
            argument(&output),
        );
        let values = device.read(&output)?;
        assert_eq!(values[0], 0);
        assert_eq!(values[1000], 632120);
        assert_eq!(values[16000], 999999);
        assert_eq!(values[16001 + 4500], 500000);
        assert!(values[..16001].windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(values[16001..].windows(2).all(|pair| pair[0] <= pair[1]));
        for (exposure, &value) in values[..16001].iter().enumerate() {
            let exact = 1.0 - (-(exposure as f64) / 1000.0).exp();
            assert!((value as f64 / 1_000_000.0 - exact).abs() <= 1.0 / 2048.0 + 0.000002);
        }
        for (index, &value) in values[16001..].iter().enumerate() {
            let score = (index as f64 - 4500.0) / 1000.0;
            let exact = 1.0 / (1.0 + (-score).exp());
            assert!((value as f64 / 1_000_000.0 - exact).abs() <= 1.0 / 2048.0 + 0.000004);
        }
        Ok(())
    }

    #[test]
    #[ignore = "optional CPU backend check"]
    fn nonlinear_table_bounds_on_cpu() -> Result<()> {
        check(Device::cpu(64)?)
    }

    #[test]
    fn nonlinear_table_bounds_on_gpu() -> Result<()> {
        check(Device::headless(64)?)
    }
}

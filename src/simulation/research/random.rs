use super::super::words::*;
use super::nonlinear::PROBABILITY_SCALE;
use cubecl::prelude::*;

pub const SHA_CONSTANTS: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];
const PREFIX: [u32; 30] = [
    98, 97, 100, 45, 101, 99, 111, 110, 47, 100, 114, 97, 119, 47, 68, 101, 116, 101, 114, 109,
    105, 110, 105, 115, 116, 105, 99, 86, 49, 0,
];
const INITIAL: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
#[derive(CubeType, Clone, Copy)]
pub struct Draw128 {
    pub first: u32,
    pub second: u32,
    pub third: u32,
    pub fourth: u32,
}

#[derive(CubeType, Clone, Copy)]
pub struct RandomKey {
    pub seed: Ticks,
    pub day: Ticks,
    pub phase: u32,
    pub actor: Ticks,
    pub purpose: u32,
    pub index: Ticks,
}

#[allow(
    clippy::manual_rotate,
    reason = "CubeCL lowers explicit shifts on both backends"
)]
#[cube]
fn rotate(value: u32, distance: u32) -> u32 {
    (value >> distance) | (value << (32 - distance))
}

#[cube]
fn reverse_bytes(value: u32) -> u32 {
    ((value & 255) << 24) | ((value & 65280) << 8) | ((value >> 8) & 65280) | (value >> 24)
}

#[cube]
fn insert_word(bytes: &mut Array<u32>, offset: usize, value: u32) {
    for index in 0..4 {
        bytes[offset + index] = (value >> (index as u32 * 8)) & 255;
    }
}

#[cube]
pub fn draw(key: RandomKey, constants: &[u32]) -> Draw128 {
    let mut bytes = Array::<u32>::new(128usize);
    for index in 0..128 {
        bytes[index] = 0;
    }
    #[unroll]
    for index in 0..30 {
        bytes[index] = comptime!(PREFIX[index]);
    }
    insert_word(&mut bytes, 30, key.seed.extract(0usize));
    insert_word(&mut bytes, 34, key.seed.extract(1usize));
    insert_word(&mut bytes, 38, key.day.extract(0usize));
    insert_word(&mut bytes, 42, key.day.extract(1usize));
    insert_word(&mut bytes, 46, key.phase);
    insert_word(&mut bytes, 50, key.actor.extract(0usize));
    insert_word(&mut bytes, 54, key.actor.extract(1usize));
    insert_word(&mut bytes, 58, key.purpose);
    insert_word(&mut bytes, 62, key.index.extract(0usize));
    insert_word(&mut bytes, 66, key.index.extract(1usize));
    bytes[70] = 128;
    bytes[126] = 2;
    bytes[127] = 48;
    let mut hash = Array::<u32>::new(8usize);
    #[unroll]
    for index in 0..8 {
        hash[index] = comptime!(INITIAL[index]);
    }
    let mut words = Array::<u32>::new(64usize);
    for block in 0..2 {
        for index in 0..16 {
            let position = block * 64 + index * 4;
            words[index] = (bytes[position] << 24)
                | (bytes[position + 1] << 16)
                | (bytes[position + 2] << 8)
                | bytes[position + 3];
        }
        for index in 16..64 {
            let first = words[index - 15];
            let second = words[index - 2];
            let sigma0 = rotate(first, 7) ^ rotate(first, 18) ^ (first >> 3);
            let sigma1 = rotate(second, 17) ^ rotate(second, 19) ^ (second >> 10);
            words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
        }
        let mut a = hash[0];
        let mut b = hash[1];
        let mut c = hash[2];
        let mut d = hash[3];
        let mut e = hash[4];
        let mut f = hash[5];
        let mut g = hash[6];
        let mut h = hash[7];
        for index in 0..64 {
            let upper = rotate(e, 6) ^ rotate(e, 11) ^ rotate(e, 25);
            let choice = (e & f) ^ ((!e) & g);
            let first = h + upper + choice + constants[index] + words[index];
            let lower = rotate(a, 2) ^ rotate(a, 13) ^ rotate(a, 22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = lower + majority;
            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }
        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }
    Draw128 {
        first: reverse_bytes(hash[0]),
        second: reverse_bytes(hash[1]),
        third: reverse_bytes(hash[2]),
        fourth: reverse_bytes(hash[3]),
    }
}

#[derive(CubeType, Clone, Copy)]
struct ThresholdDigit {
    quotient: u32,
    remainder: u32,
}

#[cube]
fn threshold_digit(remainder: u32) -> ThresholdDigit {
    let numerator = ticks(0, remainder);
    let quotient = divide(numerator, PROBABILITY_SCALE);
    ThresholdDigit {
        quotient: quotient.extract(0usize),
        remainder: minus(numerator, times(quotient, PROBABILITY_SCALE).value).extract(0usize),
    }
}

#[cube]
pub fn succeeds(probability: u32, random: Draw128) -> bool {
    let fourth = threshold_digit(probability);
    let third = threshold_digit(fourth.remainder);
    let second = threshold_digit(third.remainder);
    let first = threshold_digit(second.remainder);
    let below = random.fourth < fourth.quotient
        || (random.fourth == fourth.quotient && random.third < third.quotient)
        || (random.fourth == fourth.quotient
            && random.third == third.quotient
            && random.second < second.quotient)
        || (random.fourth == fourth.quotient
            && random.third == third.quotient
            && random.second == second.quotient
            && random.first < first.quotient);
    probability == PROBABILITY_SCALE || (probability > 0 && below)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::Device;
    use anyhow::Result;
    use sha2::{Digest, Sha256};

    #[cube(launch)]
    fn probe(input: &[u32], constants: &[u32], output: &mut [u32]) {
        let id = ABSOLUTE_POS;
        if id < input.len() / 10 {
            let base = id * 10;
            let key = RandomKey {
                seed: ticks(input[base], input[base + 1]),
                day: ticks(input[base + 2], input[base + 3]),
                phase: input[base + 4],
                actor: ticks(input[base + 5], input[base + 6]),
                purpose: input[base + 7],
                index: ticks(input[base + 8], input[base + 9]),
            };
            let random = draw(key, constants);
            output[id * 4] = random.first;
            output[id * 4 + 1] = random.second;
            output[id * 4 + 2] = random.third;
            output[id * 4 + 3] = random.fourth;
        }
    }

    #[cube(launch)]
    fn threshold_probe(input: &[u32], output: &mut [u32]) {
        let id = ABSOLUTE_POS;
        if id < input.len() / 5 {
            let base = id * 5;
            let random = Draw128 {
                first: input[base + 1],
                second: input[base + 2],
                third: input[base + 3],
                fourth: input[base + 4],
            };
            output[id] = u32::cast_from(succeeds(input[base], random));
        }
    }

    fn check(device: Device) -> Result<()> {
        let cases: [[u32; 10]; 4] = [
            [42, 0, 7, 0, 3, 12, 0, 9, 0, 0],
            [0; 10],
            [u32::MAX; 10],
            [0x87654321, 0x12345678, 9, 3, 64, 11, 7, 0x52535243, 99, 111],
        ];
        let input = device.tensor(bytemuck::cast_slice(&cases));
        let constants = device.tensor(&SHA_CONSTANTS);
        let output = device.empty(cases.len() * 4);
        let argument = |tensor: &burn_wgpu::CubeTensor| unsafe {
            BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
        };
        probe::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(device.workgroup),
            argument(&input),
            argument(&constants),
            argument(&output),
        );
        let actual = device.read(&output)?;
        for (case, actual) in cases.iter().zip(actual.chunks_exact(4)) {
            let mut hash = Sha256::new();
            hash.update(b"bad-econ/draw/DeterministicV1\0");
            case.iter().for_each(|word| hash.update(word.to_le_bytes()));
            let digest = hash.finalize();
            let expected = digest[..16]
                .chunks_exact(4)
                .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for probability in [0, 1, 333333, 500000, 999999, PROBABILITY_SCALE] {
            let threshold = match probability {
                PROBABILITY_SCALE => u128::MAX,
                p => {
                    (u128::MAX / u128::from(PROBABILITY_SCALE)) * u128::from(p)
                        + ((u128::MAX % u128::from(PROBABILITY_SCALE)) + 1) * u128::from(p)
                            / u128::from(PROBABILITY_SCALE)
                }
            };
            for random in [
                0,
                threshold.saturating_sub(1),
                threshold,
                threshold.saturating_add(1),
                u128::MAX,
            ] {
                inputs.push(probability);
                inputs.extend((0..4).map(|digit| (random >> (digit * 32)) as u32));
                expected.push(u32::from(
                    probability == PROBABILITY_SCALE || random < threshold,
                ));
            }
        }
        let input = device.tensor(&inputs);
        let output = device.empty(expected.len());
        threshold_probe::launch(
            &device.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(device.workgroup),
            argument(&input),
            argument(&output),
        );
        assert_eq!(device.read(&output)?, expected);
        Ok(())
    }

    #[test]
    fn sha256_and_128_bit_floored_thresholds_on_gpu() -> Result<()> {
        check(Device::headless(64)?)
    }
}

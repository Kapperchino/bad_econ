use super::{Device, tensor};
use anyhow::Result;
use burn::tensor::Shape;
use burn_wgpu::CubeTensor;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy)]
#[repr(u8)]
pub(super) enum Stream {
    Research = 0,
    Strikes = 1,
    Demography = 2,
}

pub(super) fn samples(
    device: &Device,
    seed: u64,
    day: u64,
    stream: Stream,
    shape: impl Into<Shape>,
) -> Result<CubeTensor> {
    let mut hash = Sha256::new();
    hash.update(b"bad-econ/cubek-random/v1");
    hash.update(seed.to_le_bytes());
    hash.update(day.to_le_bytes());
    hash.update([stream as u8]);
    let digest = hash.finalize();
    let seed = u64::from_le_bytes(digest[..8].try_into()?);
    tensor::uniform(device, shape, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn draws(device: &Device, seed: u64, day: u64, stream: Stream) -> Result<Vec<u32>> {
        let output = samples(device, seed, day, stream, [1027])?;
        let bytes = device.client.read_one(output.handle)?;
        let values = bytemuck::cast_slice::<u8, f32>(&bytes);
        assert!(values.iter().all(|value| (0.0..1.0).contains(value)));
        Ok(values.iter().map(|value| value.to_bits()).collect())
    }

    #[test]
    fn gpu_random_streams_replay_across_workgroups_and_concurrent_launches() -> Result<()> {
        let device = Device::headless(64)?;
        let expected = draws(&device, 42, 9, Stream::Research)?;
        assert_eq!(
            expected,
            draws(&Device::headless(128)?, 42, 9, Stream::Research)?
        );
        assert_ne!(expected, draws(&device, 43, 9, Stream::Research)?);
        assert_ne!(expected, draws(&device, 42, 10, Stream::Research)?);
        assert_ne!(expected, draws(&device, 42, 9, Stream::Strikes)?);
        assert_ne!(expected, draws(&device, 42, 9, Stream::Demography)?);
        std::thread::scope(|scope| -> Result<()> {
            let workers = (0..4)
                .map(|seed| {
                    let device = &device;
                    scope.spawn(move || draws(device, seed, 9, Stream::Research))
                })
                .collect::<Vec<_>>();
            for (seed, worker) in workers.into_iter().enumerate() {
                assert_eq!(
                    worker.join().expect("GPU random worker")?,
                    draws(&device, seed as u64, 9, Stream::Research)?
                );
            }
            Ok(())
        })?;
        Ok(())
    }
}

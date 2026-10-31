use super::{
    economy::{self, EconomyStage},
    scheduler::{self, SchedulerStage},
};
use anyhow::{Context, Result, anyhow};
use burn::{
    backend::Backend,
    tensor::{DType, Shape},
};
use burn_wgpu::{CubeBackend, CubeTensor, WgpuSetup, wgpu};
use cubecl::{CubeCount, CubeDim, client::Client, prelude::BufferArg};

pub struct Device {
    pub client: Client,
    pub cube: cubecl::Device,
    pub name: String,
    pub workgroup: u32,
}
impl Device {
    pub fn headless(workgroup: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .context("no GPU adapter")?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let descriptor = wgpu::DeviceDescriptor {
            label: Some("bad-econ simulation"),
            required_limits: limits,
            ..Default::default()
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))?;
        Self::import(
            WgpuSetup {
                instance,
                adapter,
                device,
                queue,
                backend: info.backend,
            },
            workgroup,
        )
    }
    pub fn import(setup: WgpuSetup, workgroup: u32) -> Result<Self> {
        match [32, 64, 128, 256].contains(&workgroup) {
            true => {
                let name = format!("{} / {:?}", setup.adapter.get_info().name, setup.backend);
                let imported =
                    burn_wgpu::try_init_device(setup, burn_wgpu::RuntimeOptions::default())
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                let cube = cubecl::Device::Wgpu(imported);
                let client = cube.client();
                Ok(Self {
                    client,
                    cube,
                    name,
                    workgroup,
                })
            }
            false => Err(anyhow!("workgroup must be 32, 64, 128, or 256")),
        }
    }
    pub fn tensor(&self, words: &[u32]) -> CubeTensor {
        let bytes = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        CubeTensor::new_contiguous(
            self.client.clone(),
            self.cube.clone(),
            Shape::new([words.len()]),
            self.client.create_from_slice(&bytes),
            DType::U32,
        )
    }
    pub fn empty(&self, words: usize) -> CubeTensor {
        CubeTensor::new_contiguous(
            self.client.clone(),
            self.cube.clone(),
            Shape::new([words]),
            self.client.empty(words * 4),
            DType::U32,
        )
    }
    pub fn read(&self, tensor: &CubeTensor) -> Result<Vec<u32>> {
        Ok(self
            .client
            .read_one(tensor.handle.clone())?
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect())
    }
    pub fn read_range(&self, tensor: &CubeTensor, start: usize, length: usize) -> Result<Vec<u32>> {
        let handle = tensor.handle.clone();
        let size = handle.size_in_used();
        let handle = handle
            .offset_start(start as u64 * 4)
            .offset_end(size - (start + length) as u64 * 4);
        Ok(self
            .client
            .read_one(handle)?
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect())
    }
    pub fn write(&self, tensor: &CubeTensor, words: &[u32]) {
        self.client.write(
            &tensor.handle,
            burn::tensor::Bytes::from_elems(
                words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
        );
    }
    pub fn sync(&self) -> Result<()> {
        pollster::block_on(self.client.sync()).map_err(Into::into)
    }
    fn count(&self, count: usize) -> CubeCount {
        let groups = (count as u32).div_ceil(self.workgroup).max(1);
        let x = groups.min(65535);
        CubeCount::Static(x, groups.div_ceil(x), 1)
    }
    pub fn cpu(workgroup: u32) -> Result<Self> {
        let cube = cubecl::Device::cpu().map_err(|error| anyhow::anyhow!("{error}"))?;
        let client = cube.client();
        match [32, 64, 128, 256].contains(&workgroup) {
            true => Ok(Self {
                client,
                cube,
                name: "CubeCL CPU / LLVM".into(),
                workgroup,
            }),
            false => Err(anyhow!("workgroup must be 32, 64, 128, or 256")),
        }
    }
}

pub trait EconomyOps: Backend {
    fn economy(device: &Device, stage: EconomyStage, count: usize, buffers: &[&CubeTensor]);
    fn schedule(device: &Device, stage: SchedulerStage, count: usize, buffers: &[&CubeTensor]);
}
impl EconomyOps for CubeBackend {
    fn economy(device: &Device, stage: EconomyStage, count: usize, buffers: &[&CubeTensor]) {
        let argument = |index: usize| unsafe {
            BufferArg::from_raw_parts(
                buffers[index].handle.clone(),
                buffers[index].meta.shape()[0],
            )
        };
        economy::economy::launch(
            &device.client,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            argument(0),
            argument(1),
            argument(2),
            argument(3),
            argument(4),
            argument(5),
            argument(6),
            argument(7),
            stage,
        );
    }
    fn schedule(device: &Device, stage: SchedulerStage, count: usize, buffers: &[&CubeTensor]) {
        let argument = |index: usize| unsafe {
            BufferArg::from_raw_parts(
                buffers[index].handle.clone(),
                buffers[index].meta.shape()[0],
            )
        };
        scheduler::schedule::launch(
            &device.client,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            argument(0),
            argument(1),
            argument(2),
            argument(3),
            argument(4),
            argument(5),
            stage,
        );
    }
}

use super::{
    economy::{self, EconomyStage},
    layout::WORD_BYTES,
    parameters::{EconomyParameters, SchedulerParameters},
    scheduler::{self, SchedulerStage},
};
use anyhow::{Context, Result, anyhow};
use burn::{
    backend::Backend,
    tensor::{DType, Shape},
};
use burn_wgpu::{CubeBackend, CubeTensor, WgpuSetup, wgpu};
use cubecl::{CubeCount, CubeDim, client::Client, prelude::BufferArg};

const WORKGROUP_SIZES: [u32; 4] = [32, 64, 128, 256];
const MAX_DISPATCH_GROUPS_PER_AXIS: u32 = u16::MAX as u32;

#[cfg(test)]
pub const TEST_WORKGROUP: u32 = 64;

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
        match WORKGROUP_SIZES.contains(&workgroup) {
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
            false => Err(anyhow!("workgroup must be one of {WORKGROUP_SIZES:?}")),
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
            self.client.empty(words * WORD_BYTES),
            DType::U32,
        )
    }
    pub fn read(&self, tensor: &CubeTensor) -> Result<Vec<u32>> {
        Ok(self
            .client
            .read_one(tensor.handle.clone())?
            .chunks_exact(WORD_BYTES)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes")))
            .collect())
    }
    pub fn read_range(&self, tensor: &CubeTensor, start: usize, length: usize) -> Result<Vec<u32>> {
        let handle = tensor.handle.clone();
        let size = handle.size_in_used();
        let handle = handle
            .offset_start((start * WORD_BYTES) as u64)
            .offset_end(size - ((start + length) * WORD_BYTES) as u64);
        Ok(self
            .client
            .read_one(handle)?
            .chunks_exact(WORD_BYTES)
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
        let x = groups.min(MAX_DISPATCH_GROUPS_PER_AXIS);
        CubeCount::Static(x, groups.div_ceil(x), 1)
    }
    pub fn cpu(workgroup: u32) -> Result<Self> {
        let cube = cubecl::Device::cpu().map_err(|error| anyhow::anyhow!("{error}"))?;
        let client = cube.client();
        match WORKGROUP_SIZES.contains(&workgroup) {
            true => Ok(Self {
                client,
                cube,
                name: "CubeCL CPU / LLVM".into(),
                workgroup,
            }),
            false => Err(anyhow!("workgroup must be one of {WORKGROUP_SIZES:?}")),
        }
    }
}

pub struct EconomyBuffers<'a> {
    pub state: &'a CubeTensor,
    pub actors: &'a CubeTensor,
    pub config: &'a CubeTensor,
    pub spatial: &'a CubeTensor,
    pub proposals: &'a CubeTensor,
    pub events: &'a CubeTensor,
    pub control: &'a CubeTensor,
}

#[derive(Clone, Copy)]
pub struct SchedulerBuffers<'a> {
    pub state: &'a CubeTensor,
    pub events: &'a CubeTensor,
    pub scratch: &'a CubeTensor,
    pub control: &'a CubeTensor,
    pub links: &'a CubeTensor,
}

pub trait EconomyOps: Backend {
    fn economy(
        device: &Device,
        stage: EconomyStage,
        count: usize,
        buffers: EconomyBuffers<'_>,
        parameters: &EconomyParameters,
    );
    fn schedule(
        device: &Device,
        stage: SchedulerStage,
        count: usize,
        buffers: SchedulerBuffers<'_>,
        parameters: SchedulerParameters,
    );
}
impl EconomyOps for CubeBackend {
    fn economy(
        device: &Device,
        stage: EconomyStage,
        count: usize,
        buffers: EconomyBuffers<'_>,
        parameters: &EconomyParameters,
    ) {
        let argument = |tensor: &CubeTensor| unsafe {
            BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
        };
        economy::economy::launch(
            &device.client,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            argument(buffers.state),
            argument(buffers.actors),
            argument(buffers.config),
            argument(buffers.spatial),
            argument(buffers.proposals),
            argument(buffers.events),
            argument(buffers.control),
            parameters.launch(),
            stage,
        );
    }
    fn schedule(
        device: &Device,
        stage: SchedulerStage,
        count: usize,
        buffers: SchedulerBuffers<'_>,
        parameters: SchedulerParameters,
    ) {
        let argument = |tensor: &CubeTensor| unsafe {
            BufferArg::from_raw_parts(tensor.handle.clone(), tensor.meta.shape()[0])
        };
        scheduler::schedule::launch(
            &device.client,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            argument(buffers.state),
            argument(buffers.events),
            argument(buffers.scratch),
            argument(buffers.control),
            argument(buffers.links),
            parameters.launch(),
            stage,
        );
    }
}

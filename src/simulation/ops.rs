use super::{
    economy::{self, EconomyStage},
    layout::WORD_BYTES,
    memory::MemoryBudget,
    parameters::{EconomyParameters, SchedulerParameters},
    scheduler::{self, SchedulerStage},
    system::System,
};
use anyhow::{Context, Result, anyhow};
use burn::{
    backend::Backend,
    tensor::{DType, Shape},
};
use burn_wgpu::{CubeBackend, CubeTensor, WgpuSetup, wgpu};
use cubecl::{CubeCount, CubeDim, client::Client};

const WORKGROUP_SIZES: [u32; 4] = [32, 64, 128, 256];
const DEDICATED_BUFFER_BYTES: usize = 1024 * 1024;
const STATE_BUFFER_BYTES: usize = 16 * 1024 * 1024;
const MAX_READBACK_WORDS: usize = 4 * 1024 * 1024;
const DIRECT_READBACK_WORDS: usize = 1024;

enum TensorAllocation {
    Pooled,
    Dedicated,
}

#[cfg(test)]
pub const TEST_WORKGROUP: u32 = 64;

#[derive(Clone)]
pub struct Device {
    pub client: Client,
    pub cube: cubecl::Device,
    pub name: String,
    pub workgroup: u32,
    pub memory_budget: MemoryBudget,
}
impl Device {
    pub fn headless(workgroup: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter")?;
        let info = adapter.get_info();
        let native = match info.backend {
            #[cfg(any(
                windows,
                target_os = "linux",
                target_os = "android",
                target_os = "freebsd"
            ))]
            wgpu::Backend::Vulkan => {
                pollster::block_on(cubecl::wgpu::vulkan::try_request_vulkan_device(&adapter))
                    .map_err(|error| anyhow!("{error}"))?
            }
            _ => None,
        };
        let (device, queue) = match native {
            Some(native) => native,
            None => {
                let descriptor = wgpu::DeviceDescriptor {
                    label: Some("bad-econ simulation"),
                    required_limits: adapter.limits(),
                    required_features: adapter.features()
                        & (wgpu::Features::TIMESTAMP_QUERY
                            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                            | wgpu::Features::SHADER_INT64
                            | wgpu::Features::SUBGROUP),
                    ..Default::default()
                };
                pollster::block_on(adapter.request_device(&descriptor))?
            }
        };
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
        match setup
            .device
            .features()
            .contains(wgpu::Features::SHADER_INT64)
        {
            true => (),
            false => Err(anyhow!(
                "CubeK integer reductions require GPU SHADER_INT64 support"
            ))?,
        }
        match WORKGROUP_SIZES.contains(&workgroup) {
            true => {
                let name = format!("{} / {:?}", setup.adapter.get_info().name, setup.backend);
                let memory_budget = MemoryBudget::gpu(&setup.adapter);
                let options = burn_wgpu::RuntimeOptions::with_tasks_max(
                    std::env::var_os("CUBECL_WGPU_MAX_TASKS")
                        .is_none()
                        .then_some(128),
                )
                .map_err(|error| anyhow!("{error}"))?;
                let imported = burn_wgpu::try_init_device(setup, options)
                    .map_err(|error| anyhow!("{error}"))?;
                let cube = cubecl::Device::Wgpu(imported);
                let client = cube.client();
                let name = format!("{name} / {}", client.name());
                Ok(Self {
                    client,
                    cube,
                    name,
                    workgroup,
                    memory_budget,
                })
            }
            false => Err(anyhow!("workgroup must be one of {WORKGROUP_SIZES:?}")),
        }
    }
    pub fn tensor(&self, words: &[u32]) -> CubeTensor {
        let allocation = match words.len() * WORD_BYTES {
            bytes if bytes >= DEDICATED_BUFFER_BYTES => TensorAllocation::Dedicated,
            _ => TensorAllocation::Pooled,
        };
        self.upload(words, allocation)
    }
    pub(super) fn dedicated_tensor(&self, words: &[u32]) -> CubeTensor {
        self.upload(words, TensorAllocation::Dedicated)
    }
    fn upload(&self, words: &[u32], allocation: TensorAllocation) -> CubeTensor {
        #[cfg(target_endian = "little")]
        let bytes: &[u8] = bytemuck::cast_slice(words);
        #[cfg(target_endian = "big")]
        let encoded = words
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        #[cfg(target_endian = "big")]
        let bytes = encoded.as_slice();
        let handle = match allocation {
            TensorAllocation::Dedicated => self
                .client
                .memory_dedicated_allocation(bytes, |bytes| self.client.create_from_slice(bytes)),
            TensorAllocation::Pooled => self.client.create_from_slice(bytes),
        };
        CubeTensor::new_contiguous(
            self.client.clone(),
            self.cube.clone(),
            Shape::new([words.len()]),
            handle,
            DType::U32,
        )
    }
    pub fn empty(&self, words: usize) -> CubeTensor {
        let handle = match words * WORD_BYTES {
            bytes if bytes >= DEDICATED_BUFFER_BYTES => self
                .client
                .memory_dedicated_allocation(bytes, |bytes| self.client.empty(bytes)),
            bytes => self.client.empty(bytes),
        };
        CubeTensor::new_contiguous(
            self.client.clone(),
            self.cube.clone(),
            Shape::new([words]),
            handle,
            DType::U32,
        )
    }
    pub(super) fn state_buffer(&self, words: usize) -> CubeTensor {
        match words * WORD_BYTES {
            bytes if bytes >= STATE_BUFFER_BYTES => {
                let capacity = bytes.next_multiple_of(STATE_BUFFER_BYTES);
                let handle = self
                    .client
                    .memory_persistent_allocation(capacity, |capacity| self.client.empty(capacity))
                    .offset_end((capacity - bytes) as u64);
                CubeTensor::new_contiguous(
                    self.client.clone(),
                    self.cube.clone(),
                    Shape::new([words]),
                    handle,
                    DType::U32,
                )
            }
            _ => self.empty(words),
        }
    }
    pub fn read(&self, tensor: &CubeTensor) -> Result<Vec<u32>> {
        self.read_range(tensor, 0, tensor.meta.shape()[0])
    }
    pub fn read_range(&self, tensor: &CubeTensor, start: usize, length: usize) -> Result<Vec<u32>> {
        let chunk = match length {
            0..=DIRECT_READBACK_WORDS => length.max(1),
            _ => (1usize << length.ilog2()).min(MAX_READBACK_WORDS),
        };
        let mut words = Vec::with_capacity(length);
        for offset in (0..length).step_by(chunk) {
            let base = offset.min(length - chunk);
            let handle = tensor.handle.clone();
            let size = handle.size_in_used();
            let handle = handle
                .offset_start(((start + base) * WORD_BYTES) as u64)
                .offset_end(size - ((start + base + chunk) * WORD_BYTES) as u64);
            let bytes = self.client.read_one(handle)?;
            words.extend(
                bytes
                    .chunks_exact(WORD_BYTES)
                    .skip(offset - base)
                    .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("four bytes"))),
            );
        }
        Ok(words)
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
    pub(super) fn reclaim_memory(&self, additional_bytes: u64) -> Result<()> {
        let reserved = self
            .client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
            .bytes_reserved;
        if reserved.saturating_add(additional_bytes) > self.memory_budget.bytes() {
            self.sync()?;
            self.client.memory_cleanup()?;
        }
        Ok(())
    }
    pub(super) fn count(&self, count: usize) -> CubeCount {
        cubecl::calculate_cube_count_elemwise(
            &self.client,
            count.max(1),
            CubeDim::new_1d(self.workgroup),
        )
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
                memory_budget: MemoryBudget::Unreported,
            }),
            false => Err(anyhow!("workgroup must be one of {WORKGROUP_SIZES:?}")),
        }
    }
}

pub struct EconomyBuffers<'a> {
    pub state: &'a CubeTensor,
    pub actors: &'a CubeTensor,
    pub config: &'a super::configuration::Configuration,
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
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        match (stage, parameters.phase) {
            (EconomyStage::Local, crate::domain::Phase::Opening) => economy::opening::launch(
                device,
                buffers.state,
                buffers.config,
                buffers.control,
                parameters.day,
            ),
            _ => economy::EconomySystem::launch(
                device,
                device.count(count),
                CubeDim::new_1d(device.workgroup),
                economy::EconomyInputLaunch {
                    state: argument(buffers.state),
                    actors: argument(buffers.actors),
                    config: buffers.config.launch(),
                    spatial: argument(buffers.spatial),
                    proposals: argument(buffers.proposals),
                    bundles: argument(buffers.events),
                    control: argument(buffers.control),
                    parameters: parameters.launch(),
                    stage,
                },
            ),
        }
    }
    fn schedule(
        device: &Device,
        stage: SchedulerStage,
        count: usize,
        buffers: SchedulerBuffers<'_>,
        parameters: SchedulerParameters,
    ) {
        let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
        scheduler::EventScheduling::launch(
            device,
            device.count(count),
            CubeDim::new_1d(device.workgroup),
            scheduler::ScheduleInputLaunch {
                balances: argument(buffers.state),
                events: argument(buffers.events),
                scratch: argument(buffers.scratch),
                control: argument(buffers.control),
                links: argument(buffers.links),
                params: parameters.launch(),
                stage,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reusable_state_buffers_preserve_logical_bounds_and_clear_stale_snapshots() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let words = STATE_BUFFER_BYTES / WORD_BYTES + 4;
        let stale = device.state_buffer(words + 4);
        device.write(&stale, &vec![u32::MAX; words + 4]);
        device.sync()?;
        let reserved = || {
            device
                .client
                .memory_report(cubecl::MemoryScope::Device)
                .streams
                .iter()
                .map(|stream| stream.pools.persistent.usage.bytes_reserved)
                .sum::<u64>()
        };
        let before = reserved();
        drop(stale);
        let target = device.state_buffer(words);
        device.sync()?;
        assert_eq!(target.handle.size_in_used(), (words * WORD_BYTES) as u64);
        assert_eq!(reserved(), before);
        let values = vec![0; words];
        let source = device.tensor(&values);
        let snapshot = crate::simulation::tensor::SnapshotCopy::new(&device, words);
        snapshot.save(&device, &source, &target);
        assert_eq!(device.read(&target)?, values);
        Ok(())
    }
}

use super::memory::MemoryBudget;
use crate::simulation::storage::layout::WORD_BYTES;
use anyhow::{Context, Result, anyhow};
use burn::tensor::{DType, Shape};
use burn_wgpu::{CubeTensor, WgpuSetup, wgpu};
use cubecl::{CubeCount, CubeDim, client::Client};

pub(in crate::simulation) const MAX_WORKGROUP_SIZE: u32 = 256;
const WORKGROUP_SIZES: [u32; 4] = [32, 64, 128, MAX_WORKGROUP_SIZE];
const DEDICATED_BUFFER_BYTES: usize = 1024 * 1024;
const STATE_BUFFER_BYTES: usize = 16 * 1024 * 1024;
const MAX_READBACK_WORDS: usize = 4 * 1024 * 1024;
const DIRECT_READBACK_WORDS: usize = 1024;

struct WorkgroupSize(u32);

impl WorkgroupSize {
    fn new(size: u32) -> Result<Self> {
        match WORKGROUP_SIZES.contains(&size) {
            true => Ok(Self(size)),
            false => Err(anyhow!("workgroup must be one of {WORKGROUP_SIZES:?}")),
        }
    }
}

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
    #[cfg(target_os = "macos")]
    pub fn headless(workgroup: u32) -> Result<Self> {
        use cubecl::metal::{MetalDevice, device::all_devices};
        use objc2_metal::{MTLDevice, MTLGPUFamily};

        let workgroup = WorkgroupSize::new(workgroup)?;
        let devices = all_devices();
        let discrete = devices
            .iter()
            .filter(|gpu| !gpu.hasUnifiedMemory())
            .enumerate()
            .map(|(index, gpu)| (gpu, MetalDevice::DiscreteGpu(index)));
        let integrated = devices
            .iter()
            .filter(|gpu| gpu.hasUnifiedMemory())
            .enumerate()
            .map(|(index, gpu)| (gpu, MetalDevice::IntegratedGpu(index)));
        let (gpu, kind) = discrete
            .chain(integrated)
            .find(|(gpu, _)| gpu.supportsFamily(MTLGPUFamily::Metal3))
            .context("no GPU supporting Metal 3")?;
        let cube = cubecl::Device::metal(kind)?;
        let memory_budget = MemoryBudget::Reported {
            available_bytes: super::memory::metal_available_memory(gpu),
        };
        Ok(Self::gpu(
            cube,
            gpu.name().to_string(),
            workgroup,
            memory_budget,
        ))
    }

    #[cfg(not(target_os = "macos"))]
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
        let workgroup = WorkgroupSize::new(workgroup)?;
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
        let name = format!("{} / {:?}", setup.adapter.get_info().name, setup.backend);
        let memory_budget = MemoryBudget::gpu(&setup.adapter);
        let options = burn_wgpu::RuntimeOptions::with_tasks_max(
            std::env::var_os("CUBECL_WGPU_MAX_TASKS")
                .is_none()
                .then_some(match setup.backend {
                    wgpu::Backend::Metal => 512,
                    _ => 128,
                }),
        )
        .map_err(|error| anyhow!("{error}"))?;
        let imported =
            burn_wgpu::try_init_device(setup, options).map_err(|error| anyhow!("{error}"))?;
        Ok(Self::gpu(
            cubecl::Device::Wgpu(imported),
            name,
            workgroup,
            memory_budget,
        ))
    }

    fn gpu(
        cube: cubecl::Device,
        name: String,
        workgroup: WorkgroupSize,
        memory_budget: MemoryBudget,
    ) -> Self {
        let client = cube.client();
        let name = format!("{name} / {}", client.name());
        Self {
            client,
            cube,
            name,
            workgroup: workgroup.0,
            memory_budget,
        }
    }
    pub fn tensor(&self, words: &[u32]) -> CubeTensor {
        let allocation = match words.len() * WORD_BYTES {
            bytes if bytes >= DEDICATED_BUFFER_BYTES => TensorAllocation::Dedicated,
            _ => TensorAllocation::Pooled,
        };
        self.upload(words, allocation)
    }
    pub(in crate::simulation) fn record_tensor<T: bytemuck::Pod>(
        &self,
        records: &[T],
    ) -> CubeTensor {
        match records {
            [] => self.tensor(bytemuck::cast_slice(&[T::zeroed()])),
            records => self.tensor(bytemuck::cast_slice(records)),
        }
    }
    pub(in crate::simulation) fn dedicated_tensor(&self, words: &[u32]) -> CubeTensor {
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
    pub(in crate::simulation) fn resize_scratch(
        &self,
        previous: &CubeTensor,
        words: usize,
    ) -> CubeTensor {
        let bytes = (words * WORD_BYTES) as u64;
        let mut handle = match previous.handle.size() >= bytes {
            true => previous.handle.clone(),
            false => {
                let capacity = words + words / 32;
                self.empty(capacity).handle
            }
        };
        handle.offset_end = Some(handle.size() - bytes);
        CubeTensor::new_contiguous(
            self.client.clone(),
            self.cube.clone(),
            Shape::new([words]),
            handle,
            DType::U32,
        )
    }
    pub(in crate::simulation) fn state_buffer(&self, words: usize) -> CubeTensor {
        match words * WORD_BYTES {
            bytes if (STATE_BUFFER_BYTES..256 * 1024 * 1024).contains(&bytes) => {
                let granularity = (bytes / 16).next_power_of_two().max(STATE_BUFFER_BYTES);
                let memory = &self.client.properties().memory;
                let limit = memory.max_page_size / memory.alignment * memory.alignment;
                let capacity = bytes
                    .next_multiple_of(granularity)
                    .min(limit as usize)
                    .max(bytes);
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
    fn logical_words(tensor: &CubeTensor) -> usize {
        tensor.meta.num_elements() * tensor.elem_size() / WORD_BYTES
    }
    pub fn read(&self, tensor: &CubeTensor) -> Result<Vec<u32>> {
        self.read_range(tensor, 0, Self::logical_words(tensor))
    }
    pub fn read_range(&self, tensor: &CubeTensor, start: usize, length: usize) -> Result<Vec<u32>> {
        let available =
            Self::logical_words(tensor).min(tensor.handle.size_in_used() as usize / WORD_BYTES);
        let end = start
            .checked_add(length)
            .context("readback range overflow")?;
        if end > available {
            Err(anyhow!(
                "readback range {start}..{end} exceeds {available} words"
            ))?;
        }
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
        assert!(
            words.len() <= Self::logical_words(tensor)
                && size_of_val(words) as u64 <= tensor.handle.size_in_used(),
            "write exceeds GPU tensor bounds"
        );
        #[cfg(target_endian = "little")]
        let bytes = burn::tensor::Bytes::from_elems(words.to_vec());
        #[cfg(target_endian = "big")]
        let bytes = burn::tensor::Bytes::from_elems(
            words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        self.client.write(&tensor.handle, bytes);
    }
    pub fn sync(&self) -> Result<()> {
        pollster::block_on(self.client.sync()).map_err(Into::into)
    }
    pub(in crate::simulation) fn memory_usage(&self) -> cubecl::MemoryUsage {
        self.client
            .memory_report(cubecl::MemoryScope::Device)
            .usage()
    }

    pub(in crate::simulation) fn reclaim_memory(
        &self,
        additional_bytes: u64,
    ) -> Result<cubecl::MemoryUsage> {
        let usage = self.memory_usage();
        match usage.bytes_reserved.saturating_add(additional_bytes) <= self.memory_budget.bytes() {
            true => Ok(usage),
            false => {
                self.sync()?;
                self.client.memory_cleanup()?;
                Ok(self.memory_usage())
            }
        }
    }
    pub(in crate::simulation) fn count(&self, count: usize) -> CubeCount {
        cubecl::calculate_cube_count_elemwise(
            &self.client,
            count.max(1),
            CubeDim::new_1d(self.workgroup),
        )
    }
    pub fn cpu(workgroup: u32) -> Result<Self> {
        let workgroup = WorkgroupSize::new(workgroup)?;
        let cube = cubecl::Device::cpu()?;
        Ok(Self {
            client: cube.client(),
            cube,
            name: "CubeCL CPU / LLVM".into(),
            workgroup: workgroup.0,
            memory_budget: MemoryBudget::Unreported,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readback_checks_word_ranges_before_slicing_gpu_buffers() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let buffer = device.tensor(&[10, 20, 30]);
        let matrix = super::super::tensor::reshape(&buffer, [1, 3]);
        assert_eq!(device.read(&matrix)?, [10, 20, 30]);
        assert_eq!(device.read_range(&buffer, 1, 2)?, [20, 30]);
        let empty = super::super::tensor::empty(&device, [2, 0]);
        assert!(device.read(&empty)?.is_empty());
        assert!(device.read_range(&empty, 0, 1).is_err());
        assert!(device.read_range(&buffer, 3, 0)?.is_empty());
        assert!(device.read_range(&buffer, 2, 2).is_err());
        assert!(device.read_range(&buffer, usize::MAX, 1).is_err());
        Ok(())
    }

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
        let snapshot = crate::simulation::gpu::tensor::SnapshotCopy::new(&device, words);
        snapshot.save(&device, &source, &target);
        assert_eq!(device.read(&target)?, values);
        Ok(())
    }
}

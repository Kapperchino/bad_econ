use burn_wgpu::wgpu;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(tag = "source", rename_all = "snake_case")]
pub enum MemoryBudget {
    Reported { available_bytes: u64 },
    Dedicated { available_bytes: u64 },
    Unreported,
}

impl MemoryBudget {
    pub(super) fn gpu(adapter: &wgpu::Adapter) -> Self {
        available_memory(adapter).unwrap_or(Self::Unreported)
    }

    fn from_driver(device_type: wgpu::DeviceType, available_bytes: u64) -> Self {
        match device_type {
            wgpu::DeviceType::DiscreteGpu => Self::Dedicated { available_bytes },
            _ => Self::Reported { available_bytes },
        }
    }

    pub fn bytes(self) -> u64 {
        match self {
            Self::Reported { available_bytes } => available_bytes / 2,
            Self::Dedicated { available_bytes } => available_bytes,
            Self::Unreported => 4 * 1024 * 1024 * 1024,
        }
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn metal_available_memory(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> u64 {
    use objc2_metal::MTLDevice;

    let available = device
        .recommendedMaxWorkingSetSize()
        .saturating_sub(device.currentAllocatedSize() as u64);
    match device.hasUnifiedMemory() {
        true => {
            let mut system = sysinfo::System::new();
            system.refresh_memory();
            available.min(system.available_memory())
        }
        false => available,
    }
}

fn available_memory(adapter: &wgpu::Adapter) -> Option<MemoryBudget> {
    let info = adapter.get_info();
    match info.backend {
        #[cfg(target_vendor = "apple")]
        wgpu::Backend::Metal => {
            unsafe { adapter.as_hal::<wgpu::hal::api::Metal>() }.map(|adapter| {
                MemoryBudget::Reported {
                    available_bytes: metal_available_memory(adapter.raw_device()),
                }
            })
        }
        #[cfg(any(
            windows,
            target_os = "linux",
            target_os = "android",
            target_os = "freebsd",
            target_os = "macos"
        ))]
        wgpu::Backend::Vulkan => vulkan_available_memory(adapter, info.device_type),
        #[cfg(windows)]
        wgpu::Backend::Dx12 => unsafe { adapter.as_hal::<wgpu::hal::api::Dx12>() }
            .and_then(|adapter| {
                adapter
                    .raw_adapter()
                    .query_video_memory_info(
                        windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                    )
                    .ok()
            })
            .map(|memory| {
                MemoryBudget::from_driver(
                    info.device_type,
                    memory.Budget.saturating_sub(memory.CurrentUsage),
                )
            }),
        _ => None,
    }
}

#[cfg(any(
    windows,
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "macos"
))]
fn vulkan_available_memory(
    adapter: &wgpu::Adapter,
    device_type: wgpu::DeviceType,
) -> Option<MemoryBudget> {
    use ash::vk;

    enum MemoryReport {
        Budget,
        Capacity,
    }

    let adapter = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }?;
    let shared = adapter.shared_instance();
    let instance = shared.raw_instance();
    let physical = adapter.raw_physical_device();
    let extensions = unsafe { instance.enumerate_device_extension_properties(physical) }.ok()?;
    let budget_support = extensions.iter().any(|extension| {
        extension.extension_name_as_c_str().ok() == Some(ash::ext::memory_budget::NAME)
    });
    let report = match (
        budget_support,
        shared.instance_api_version() >= vk::API_VERSION_1_1,
    ) {
        (true, true) => MemoryReport::Budget,
        _ => MemoryReport::Capacity,
    };
    let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    let memory = match report {
        MemoryReport::Budget => {
            let mut properties =
                vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
            unsafe { instance.get_physical_device_memory_properties2(physical, &mut properties) };
            properties.memory_properties
        }
        MemoryReport::Capacity => unsafe {
            instance.get_physical_device_memory_properties(physical)
        },
    };
    memory.memory_heaps[..memory.memory_heap_count as usize]
        .iter()
        .enumerate()
        .filter(|(_, heap)| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|(index, heap)| match report {
            MemoryReport::Budget => budget.heap_budget[index]
                .min(heap.size)
                .saturating_sub(budget.heap_usage[index]),
            MemoryReport::Capacity => heap.size,
        })
        .max()
        .map(|available_bytes| match report {
            MemoryReport::Budget => MemoryBudget::from_driver(device_type, available_bytes),
            MemoryReport::Capacity => MemoryBudget::Reported { available_bytes },
        })
}

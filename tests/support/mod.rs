use anyhow::Result;
use bad_econ::simulation::Device;

pub fn device(workgroup: u32) -> Result<Device> {
    Device::headless(workgroup)
}

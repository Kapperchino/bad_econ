use anyhow::{Result, anyhow};
use bad_econ::simulation::Device;
use std::sync::OnceLock;

pub fn device(workgroup: u32) -> Result<Device> {
    static DEVICE: OnceLock<Result<Device, String>> = OnceLock::new();
    let device = DEVICE
        .get_or_init(|| Device::headless(64).map_err(|error| error.to_string()))
        .as_ref()
        .map_err(|error| anyhow!(error.clone()))?;
    Ok(Device {
        workgroup,
        ..device.clone()
    })
}

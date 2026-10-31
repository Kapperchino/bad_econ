use anyhow::{Result, anyhow};
use bad_econ::simulation::Device;

pub fn device(workgroup: u32) -> Result<Device> {
    match std::env::var("BAD_ECON_TEST_BACKEND").as_deref() {
        Ok("gpu") | Err(std::env::VarError::NotPresent) => Device::headless(workgroup),
        Ok("cpu") => Device::cpu(workgroup),
        _ => Err(anyhow!("BAD_ECON_TEST_BACKEND must be cpu or gpu")),
    }
}

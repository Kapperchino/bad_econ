use anyhow::{Result, anyhow};
use bad_econ::simulation::Device;
use std::{
    collections::{HashMap, hash_map::Entry},
    sync::{Mutex, OnceLock},
};

pub fn device(workgroup: u32) -> Result<Device> {
    static DEVICES: OnceLock<Mutex<HashMap<u32, Device>>> = OnceLock::new();
    let mut devices = DEVICES
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| anyhow!("test GPU device cache poisoned"))?;
    match devices.entry(workgroup) {
        Entry::Occupied(entry) => Ok(entry.get().clone()),
        Entry::Vacant(entry) => Ok(entry.insert(Device::headless(workgroup)?).clone()),
    }
}

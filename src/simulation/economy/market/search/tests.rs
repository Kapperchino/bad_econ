use super::*;
use crate::{
    domain::{
        Experiment, FirmSeed, HouseholdSeed, Phase, Population, Position, Scenario, State,
        Technique, enterprise::HouseholdProduction,
    },
    simulation::{Market, Simulation, Trace, encode, parallel::ScanMode, tests::device},
};
use anyhow::Result;
use burn_wgpu::CubeTensor;

mod consumers;
mod delivery;
mod fixture;
mod matching;
mod preparation;
mod suppliers;

use fixture::*;

pub mod boundary;
mod systems;

use super::super::{Device, system::System};
use anyhow::{Context, Result, anyhow};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};
use systems::*;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct CapabilityEdge {
    pub prerequisite: u32,
    pub dependent: u32,
}

#[derive(Clone, Debug)]
pub struct CapabilityGraph {
    nodes: usize,
    edges: Vec<CapabilityEdge>,
}

impl CapabilityGraph {
    pub fn new(nodes: usize, mut edges: Vec<CapabilityEdge>) -> Result<Self> {
        let admitted = nodes <= 1_000_000
            && edges.len() <= 10_000_000
            && edges
                .iter()
                .all(|edge| edge.prerequisite < nodes as u32 && edge.dependent < nodes as u32);
        match admitted {
            true => {
                edges.sort_unstable();
                edges.dedup();
                Ok(Self { nodes, edges })
            }
            false => Err(anyhow!("capability graph exceeds index or size bounds")),
        }
    }

    pub fn closure(&self, device: &Device, seeds: &[u32]) -> Result<CapabilityClosure> {
        self.compute(
            device,
            seeds,
            &vec![0; self.nodes],
            ClosureDirection::Forward,
        )
    }

    pub fn missing(
        &self,
        device: &Device,
        targets: &[u32],
        satisfied: &[u32],
    ) -> Result<CapabilityClosure> {
        self.compute(device, targets, satisfied, ClosureDirection::Missing)
    }

    fn compute(
        &self,
        device: &Device,
        seeds: &[u32],
        satisfied: &[u32],
        direction: ClosureDirection,
    ) -> Result<CapabilityClosure> {
        match seeds.len() == self.nodes && satisfied.len() == self.nodes {
            false => Err(anyhow!("capability seed/mask shape mismatch")),
            true => {
                let mut adjacency = vec![Vec::new(); self.nodes];
                for edge in &self.edges {
                    match direction {
                        ClosureDirection::Forward => {
                            adjacency[edge.dependent as usize].push(edge.prerequisite)
                        }
                        ClosureDirection::Missing => {
                            adjacency[edge.prerequisite as usize].push(edge.dependent)
                        }
                    }
                }
                let mut offsets = vec![0u32];
                let mut sources = Vec::new();
                for neighbors in &adjacency {
                    sources.extend(neighbors);
                    offsets.push(
                        u32::try_from(sources.len())
                            .context("capability edge capacity exceeded")?,
                    );
                }
                let seed = device.tensor(&padded(seeds));
                let mask = device.tensor(&padded(satisfied));
                let offsets = device.tensor(&offsets);
                let sources = device.tensor(&padded(&sources));
                let mut current = device.empty(self.nodes.max(1));
                let mut next = device.empty(self.nodes.max(1));
                let changed = device.tensor(&[0]);
                let argument = |tensor: &CubeTensor| tensor.clone().into_buffer_arg();
                ClosureInitialization::launch(
                    device,
                    device.count(self.nodes),
                    CubeDim::new_1d(device.workgroup),
                    ClosureInitializationInputLaunch {
                        seed: argument(&seed),
                        mask: argument(&mask),
                        output: argument(&current),
                        nodes: self.nodes as u32,
                        direction,
                    },
                );
                let mut round = 0;
                let mut progress = ClosureProgress::Propagating;
                while round < self.nodes.saturating_sub(1)
                    && progress == ClosureProgress::Propagating
                {
                    device.write(&changed, &[0]);
                    ClosurePropagation::launch(
                        device,
                        device.count(self.nodes),
                        CubeDim::new_1d(device.workgroup),
                        ClosurePropagationInputLaunch {
                            seed: argument(&seed),
                            mask: argument(&mask),
                            offsets: argument(&offsets),
                            sources: argument(&sources),
                            previous: argument(&current),
                            next: argument(&next),
                            changed: argument(&changed),
                            nodes: self.nodes as u32,
                            direction,
                        },
                    );
                    std::mem::swap(&mut current, &mut next);
                    progress = match device.read(&changed)?[0] {
                        0 => ClosureProgress::Converged,
                        _ => ClosureProgress::Propagating,
                    };
                    round += 1;
                }
                Ok(CapabilityClosure {
                    values: current,
                    nodes: self.nodes,
                })
            }
        }
    }
}

fn padded(values: &[u32]) -> Vec<u32> {
    match values {
        [] => vec![0],
        values => values.to_vec(),
    }
}

pub struct CapabilityClosure {
    values: CubeTensor,
    nodes: usize,
}

impl CapabilityClosure {
    pub fn download(&self, device: &Device) -> Result<Vec<u32>> {
        Ok(device.read(&self.values)?[..self.nodes].to_vec())
    }

    pub fn tensor(&self) -> &CubeTensor {
        &self.values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ClosureDirection {
    Forward,
    Missing,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ClosureProgress {
    Propagating,
    Converged,
}

#[cube]
fn initialize(
    seed: &[u32],
    mask: &[u32],
    output: &mut [u32],
    nodes: u32,
    #[comptime] direction: ClosureDirection,
) {
    let id = ABSOLUTE_POS;
    if id < nodes as usize {
        output[id] = match direction {
            ClosureDirection::Forward => seed[id],
            ClosureDirection::Missing => seed[id] & !mask[id],
        };
    }
}

#[cube]
fn propagate(
    seed: &[u32],
    mask: &[u32],
    offsets: &[u32],
    sources: &[u32],
    previous: &[u32],
    next: &mut [u32],
    changed: &[Atomic<u32>],
    nodes: u32,
    #[comptime] direction: ClosureDirection,
) {
    let id = ABSOLUTE_POS;
    if id < nodes as usize {
        let mut labels = seed[id];
        for edge in offsets[id]..offsets[id + 1] {
            labels |= previous[sources[edge as usize] as usize];
        }
        next[id] = match direction {
            ClosureDirection::Forward => labels,
            ClosureDirection::Missing => labels & !mask[id],
        };
        if next[id] != previous[id] {
            changed[0].fetch_or(1);
        }
    }
}

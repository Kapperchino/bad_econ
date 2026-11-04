use super::{CapabilityClosure, CapabilityGraph, ClosureProgress, padded};
use crate::simulation::{
    Device,
    analytics::SparseMatrix,
    pass::GpuPass,
    system::{System, gpu_passes},
};
use anyhow::{Result, anyhow};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;

const MAX_BOUNDARY_NODES: usize = 4096;

pub struct DeviceCapabilityGraph {
    adjacency: CubeTensor,
    labels: CubeTensor,
    nodes: usize,
    row_words: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryObservation {
    pub nodes: usize,
    pub row_words: usize,
    pub adjacency: Vec<u32>,
    pub labels: Vec<u32>,
}

impl CapabilityGraph {
    pub fn upload(&self, device: &Device, labels: &[u32]) -> Result<DeviceCapabilityGraph> {
        let admitted = self.nodes <= MAX_BOUNDARY_NODES && labels.len() == self.nodes;
        match admitted {
            false => Err(anyhow!(
                "packed capability graph exceeds shape or node bounds"
            )),
            true => {
                let row_words = self.nodes.div_ceil(32);
                let mut adjacency = vec![0; self.nodes * row_words];
                for edge in &self.edges {
                    adjacency
                        [edge.prerequisite as usize * row_words + edge.dependent as usize / 32] |=
                        1 << (edge.dependent % 32);
                }
                Ok(DeviceCapabilityGraph {
                    adjacency: device.tensor(&padded(&adjacency)),
                    labels: device.tensor(&padded(labels)),
                    nodes: self.nodes,
                    row_words,
                })
            }
        }
    }
}

impl DeviceCapabilityGraph {
    pub fn support(device: &Device, matrix: &SparseMatrix, labels: &[u32]) -> Result<Self> {
        let admitted = matrix.rows() == matrix.columns()
            && matrix.rows() <= MAX_BOUNDARY_NODES
            && labels.len() == matrix.rows();
        match admitted {
            false => Err(anyhow!(
                "matrix support requires a square bounded capability graph"
            )),
            true => {
                let nodes = matrix.rows();
                let row_words = nodes.div_ceil(32);
                let records = matrix
                    .entries()
                    .iter()
                    .flat_map(|entry| [entry.row as u32, entry.column as u32, entry.numerator])
                    .collect::<Vec<_>>();
                let input = device.tensor(&padded(&records));
                let adjacency = device.tensor(&vec![0; (nodes * row_words).max(1)]);
                PositiveSupport::launch(
                    device,
                    count(device, matrix.entries().len()),
                    CubeDim::new_1d(device.workgroup),
                    PositiveSupportInputLaunch {
                        input: argument(&input),
                        output: argument(&adjacency),
                        entries: matrix.entries().len() as u32,
                        row_words: row_words as u32,
                    },
                );
                Ok(Self {
                    adjacency,
                    labels: device.tensor(&padded(labels)),
                    nodes,
                    row_words,
                })
            }
        }
    }

    pub fn eliminate(&self, device: &Device, retained: usize) -> Result<Self> {
        match retained <= self.nodes {
            false => Err(anyhow!(
                "retained capability boundary exceeds graph dimension"
            )),
            true => {
                let adjacency = device.empty((self.nodes * self.row_words).max(1));
                let labels = device.empty(self.nodes.max(1));
                crate::simulation::tensor::copy(device, &self.adjacency, &adjacency);
                crate::simulation::tensor::copy(device, &self.labels, &labels);
                for pivot in retained..self.nodes {
                    BoundaryElimination::launch(
                        device,
                        count(device, self.nodes),
                        CubeDim::new_1d(device.workgroup),
                        BoundaryEliminationInputLaunch {
                            adjacency: argument(&adjacency),
                            labels: argument(&labels),
                            nodes: self.nodes as u32,
                            row_words: self.row_words as u32,
                            pivot: pivot as u32,
                        },
                    );
                }
                let row_words = retained.div_ceil(32);
                let result = Self {
                    adjacency: device.empty((retained * row_words).max(1)),
                    labels: device.empty(retained.max(1)),
                    nodes: retained,
                    row_words,
                };
                BoundaryProjection::launch(
                    device,
                    count(device, retained),
                    CubeDim::new_1d(device.workgroup),
                    BoundaryProjectionInputLaunch {
                        input: argument(&adjacency),
                        labels: argument(&labels),
                        output: argument(&result.adjacency),
                        reduced_labels: argument(&result.labels),
                        old_stride: self.row_words as u32,
                        nodes: retained as u32,
                        row_words: row_words as u32,
                    },
                );
                Ok(result)
            }
        }
    }

    pub fn closure(&self, device: &Device) -> Result<CapabilityClosure> {
        let labels = device.empty(self.nodes.max(1));
        crate::simulation::tensor::copy(device, &self.labels, &labels);
        let changed = device.tensor(&[0]);
        let mut round = 0;
        let mut progress = ClosureProgress::Propagating;
        while round < self.nodes.saturating_sub(1) && progress == ClosureProgress::Propagating {
            device.write(&changed, &[0]);
            BoundaryPropagation::launch(
                device,
                count(device, self.nodes),
                CubeDim::new_1d(device.workgroup),
                BoundaryPropagationInputLaunch {
                    adjacency: argument(&self.adjacency),
                    labels: argument(&labels),
                    changed: argument(&changed),
                    nodes: self.nodes as u32,
                    row_words: self.row_words as u32,
                },
            );
            progress = match device.read(&changed)?[0] {
                0 => ClosureProgress::Converged,
                _ => ClosureProgress::Propagating,
            };
            round += 1;
        }
        Ok(CapabilityClosure {
            values: labels,
            nodes: self.nodes,
        })
    }

    pub fn download(&self, device: &Device) -> Result<BoundaryObservation> {
        Ok(BoundaryObservation {
            nodes: self.nodes,
            row_words: self.row_words,
            adjacency: device.read(&self.adjacency)?[..self.nodes * self.row_words].to_vec(),
            labels: device.read(&self.labels)?[..self.nodes].to_vec(),
        })
    }
}

fn argument(tensor: &CubeTensor) -> BufferArg {
    tensor.clone().into_buffer_arg()
}

fn count(device: &Device, items: usize) -> CubeCount {
    device.count(items)
}

#[cube]
fn positive_support(input: &[u32], output: &[Atomic<u32>], entries: u32, row_words: u32) {
    let id = ABSOLUTE_POS;
    if id < entries as usize && input[id * 3 + 2] > 0 {
        let row = input[id * 3];
        let column = input[id * 3 + 1];
        output[(row * row_words + column / 32) as usize].fetch_or(1u32 << (column % 32));
    }
}

#[cube]
fn eliminate(adjacency: &mut [u32], labels: &mut [u32], nodes: u32, row_words: u32, pivot: u32) {
    let id = ABSOLUTE_POS;
    if id < nodes as usize && id != pivot as usize {
        let adjacent =
            adjacency[id * row_words as usize + (pivot / 32) as usize] & (1u32 << (pivot % 32));
        if adjacent != 0 {
            for word in 0..row_words as usize {
                adjacency[id * row_words as usize + word] |=
                    adjacency[pivot as usize * row_words as usize + word];
            }
        }
        if adjacency[pivot as usize * row_words as usize + id / 32] & (1u32 << (id as u32 % 32))
            != 0
        {
            labels[id] |= labels[pivot as usize];
        }
    }
}

#[cube]
fn project(
    input: &[u32],
    labels: &[u32],
    output: &mut [u32],
    reduced_labels: &mut [u32],
    old_stride: u32,
    nodes: u32,
    row_words: u32,
) {
    let id = ABSOLUTE_POS;
    if id < nodes as usize {
        reduced_labels[id] = labels[id];
        for word in 0..row_words {
            let mut value = input[id * old_stride as usize + word as usize];
            if word + 1 == row_words && nodes & 31 != 0 {
                value &= (1u32 << (nodes % 32)) - 1;
            }
            output[id * row_words as usize + word as usize] = value;
        }
    }
}

#[cube]
fn propagate(
    adjacency: &[u32],
    labels: &[Atomic<u32>],
    changed: &[Atomic<u32>],
    nodes: u32,
    row_words: u32,
) {
    let id = ABSOLUTE_POS;
    if id < nodes as usize {
        let current = labels[id].load();
        if current != 0 {
            for word in 0..row_words {
                let neighbors = adjacency[id * row_words as usize + word as usize];
                if neighbors != 0 {
                    for bit in 0..32u32 {
                        let target = word * 32 + bit;
                        if target < nodes && neighbors & (1u32 << bit) != 0 {
                            let previous = labels[target as usize].fetch_or(current);
                            if current & !previous != 0 {
                                changed[0].fetch_or(1);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct PositiveSupportInput {
    pub(super) input: Box<[u32]>,
    pub(super) output: Box<[Atomic<u32>]>,
    pub(super) entries: u32,
    pub(super) row_words: u32,
}

pub(super) struct PositiveSupport;

pub(super) struct CollectPositiveSupport;

impl System for PositiveSupport {
    type Input = PositiveSupportInput;
    type Passes = gpu_passes![CollectPositiveSupport];
}

#[cube]
impl GpuPass<PositiveSupportInput> for CollectPositiveSupport {
    type Output = ();

    fn execute(input: PositiveSupportInput) {
        positive_support(&input.input, &input.output, input.entries, input.row_words);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct BoundaryEliminationInput {
    pub(super) adjacency: Box<[u32]>,
    pub(super) labels: Box<[u32]>,
    pub(super) nodes: u32,
    pub(super) row_words: u32,
    pub(super) pivot: u32,
}

pub(super) struct BoundaryElimination;

pub(super) struct EliminateNode;

impl System for BoundaryElimination {
    type Input = BoundaryEliminationInput;
    type Passes = gpu_passes![EliminateNode];
}

#[cube]
impl GpuPass<BoundaryEliminationInput> for EliminateNode {
    type Output = ();

    fn execute(mut input: BoundaryEliminationInput) {
        eliminate(
            &mut input.adjacency,
            &mut input.labels,
            input.nodes,
            input.row_words,
            input.pivot,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct BoundaryProjectionInput {
    pub(super) input: Box<[u32]>,
    pub(super) labels: Box<[u32]>,
    pub(super) output: Box<[u32]>,
    pub(super) reduced_labels: Box<[u32]>,
    pub(super) old_stride: u32,
    pub(super) nodes: u32,
    pub(super) row_words: u32,
}

pub(super) struct BoundaryProjection;

pub(super) struct ProjectBoundary;

impl System for BoundaryProjection {
    type Input = BoundaryProjectionInput;
    type Passes = gpu_passes![ProjectBoundary];
}

#[cube]
impl GpuPass<BoundaryProjectionInput> for ProjectBoundary {
    type Output = ();

    fn execute(mut input: BoundaryProjectionInput) {
        project(
            &input.input,
            &input.labels,
            &mut input.output,
            &mut input.reduced_labels,
            input.old_stride,
            input.nodes,
            input.row_words,
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct BoundaryPropagationInput {
    pub(super) adjacency: Box<[u32]>,
    pub(super) labels: Box<[Atomic<u32>]>,
    pub(super) changed: Box<[Atomic<u32>]>,
    pub(super) nodes: u32,
    pub(super) row_words: u32,
}

pub(super) struct BoundaryPropagation;

pub(super) struct PropagateBoundary;

impl System for BoundaryPropagation {
    type Input = BoundaryPropagationInput;
    type Passes = gpu_passes![PropagateBoundary];
}

#[cube]
impl GpuPass<BoundaryPropagationInput> for PropagateBoundary {
    type Output = ();

    fn execute(input: BoundaryPropagationInput) {
        propagate(
            &input.adjacency,
            &input.labels,
            &input.changed,
            input.nodes,
            input.row_words,
        );
    }
}

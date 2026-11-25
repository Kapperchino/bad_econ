use crate::simulation::{
    Device,
    pass::GpuPass,
    system::{AsBufferArg, System, gpu_passes},
};
use anyhow::{Result, anyhow};
use burn::tensor::{DType, Shape};
use burn_wgpu::CubeTensor;
use cubecl::prelude::*;
use cubek::{
    matmul::{definition::MatmulElems, strategy::Strategy},
    std::InputBinding,
};

pub(in crate::simulation) const SNAPSHOT_PARTS: usize = 4;
pub(in crate::simulation) const SNAPSHOT_BLOCK_WORDS: usize = SNAPSHOT_PARTS * 32 * 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(in crate::simulation) enum RingMaskState {
    Empty,
    Unknown,
}

impl Device {
    pub(in crate::simulation) fn clear_except_fault(&self, tensor: &CubeTensor) {
        clear_phase_counters::launch(
            &self.client,
            self.count(tensor.meta.num_elements()),
            CubeDim::new_1d(self.workgroup),
            tensor.buffer_arg(),
        );
    }

    pub(in crate::simulation) fn clear(&self, tensor: &CubeTensor) {
        clear_words::launch(
            &self.client,
            self.count(tensor.meta.num_elements()),
            CubeDim::new_1d(self.workgroup),
            tensor.buffer_arg(),
        );
    }
}

#[cube(launch)]
fn clear_phase_counters(output: &mut [u32]) {
    if ABSOLUTE_POS < output.len()
        && ABSOLUTE_POS != comptime!(crate::simulation::storage::flags::FAULT_WORD)
    {
        output[ABSOLUTE_POS] = 0;
    }
}

#[cube(launch)]
fn clear_words(output: &mut [u32]) {
    if ABSOLUTE_POS < output.len() {
        output[ABSOLUTE_POS] = 0;
    }
}

pub(in crate::simulation) fn reset_ring_mask(
    device: &Device,
    mask: &CubeTensor,
    state: RingMaskState,
) {
    RingMaskReset::launch_elements(
        device,
        mask.meta.num_elements(),
        RingMaskInputLaunch {
            mask: mask.buffer_arg(),
            state,
        },
    );
}

#[derive(CubeType, CubeLaunch)]
struct RingMaskInput {
    mask: Box<[Atomic<u32>]>,
    #[cube(comptime)]
    state: RingMaskState,
}

struct RingMaskReset;
struct ResetRingMask;

impl System for RingMaskReset {
    type Input = RingMaskInput;
    type Passes = gpu_passes![ResetRingMask];
}

#[cube]
impl GpuPass<RingMaskInput> for ResetRingMask {
    type Output = ();

    fn execute(input: RingMaskInput) {
        if ABSOLUTE_POS < input.mask.len() {
            input.mask[ABSOLUTE_POS].store(comptime!(match input.state {
                RingMaskState::Empty => 0,
                RingMaskState::Unknown => 1,
            }));
        }
    }
}

pub(in crate::simulation) fn state_values(
    values: &[u64],
    actors: usize,
    stride: usize,
) -> std::borrow::Cow<'_, [u64]> {
    match stride == actors {
        true => std::borrow::Cow::Borrowed(values),
        false => std::borrow::Cow::Owned(
            values
                .chunks_exact(actors)
                .flat_map(|field| {
                    field
                        .iter()
                        .copied()
                        .chain(std::iter::repeat_n(0, stride - actors))
                })
                .collect(),
        ),
    }
}

pub(in crate::simulation) fn vector_width(device: &Device, elements: usize) -> usize {
    device
        .client
        .io_optimized_vector_sizes(size_of::<f32>())
        .filter(|&width| width <= 4 && elements.max(1).is_multiple_of(width))
        .max()
        .unwrap_or(1)
}

pub(in crate::simulation) fn floats(
    device: &Device,
    shape: impl Into<Shape>,
    values: &[f32],
) -> CubeTensor {
    let shape = shape.into();
    assert_eq!(
        shape.num_elements(),
        values.len(),
        "float upload shape mismatch"
    );
    let handle = match values.len() {
        0 => device.client.empty(size_of::<f32>()),
        _ => device
            .client
            .create_from_slice(bytemuck::cast_slice(values)),
    };
    CubeTensor::new_contiguous(
        device.client.clone(),
        device.cube.clone(),
        shape,
        handle,
        DType::F32,
    )
}

pub(in crate::simulation) fn empty(device: &Device, shape: impl Into<Shape>) -> CubeTensor {
    let shape = shape.into();
    let handle = device
        .client
        .empty(shape.num_elements().max(1) * size_of::<f32>());
    CubeTensor::new_contiguous(
        device.client.clone(),
        device.cube.clone(),
        shape,
        handle,
        DType::F32,
    )
}

pub(in crate::simulation) fn reshape(tensor: &CubeTensor, shape: impl Into<Shape>) -> CubeTensor {
    let shape = shape.into();
    assert_eq!(
        shape.num_elements(),
        tensor.meta.num_elements(),
        "reshape size mismatch"
    );
    assert!(
        tensor.is_contiguous() && !tensor.meta.is_tiled(),
        "reshape requires contiguous rows"
    );
    CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        shape,
        tensor.handle.clone(),
        tensor.dtype,
    )
}

pub(in crate::simulation) fn matmul(
    device: &Device,
    left: &CubeTensor,
    right: &CubeTensor,
    output: &CubeTensor,
) -> Result<()> {
    let inner = left.meta.shape()[left.meta.num_dims() - 1];
    match output.meta.num_elements() {
        0 => Ok(()),
        _ if inner == 0 => {
            device.client.write(
                &output.handle,
                burn::tensor::Bytes::from_elems(vec![0.0f32; output.meta.num_elements()]),
            );
            Ok(())
        }
        _ => cubek::matmul::launch::launch_ref(
            &Strategy::Auto,
            &device.client,
            InputBinding::new(left.clone().binding(), f32::elem_type_native()),
            InputBinding::new(right.clone().binding(), f32::elem_type_native()),
            output.clone().binding(),
            &mut MatmulElems::from_single_dtype(f32::elem_type_native()),
        )
        .map_err(|error| anyhow!("Cubek matrix multiplication: {error:?}")),
    }
}

pub(in crate::simulation) fn uniform(
    device: &Device,
    shape: impl Into<Shape>,
    seed: u64,
) -> Result<CubeTensor> {
    let output = empty(device, shape);
    if output.meta.num_elements() > 0 {
        cubek::random::random_uniform(
            &device.client,
            0.0,
            1.0,
            output.clone().binding(),
            f32::elem_type_native(),
            seed,
        )?;
    }
    Ok(output)
}

pub(in crate::simulation) fn same_buffer(recorded: &CubeTensor, current: &CubeTensor) -> bool {
    (
        recorded.handle.memory.descriptor(),
        recorded.handle.service,
        recorded.handle.offset_start,
        recorded.handle.offset_end,
        recorded.handle.size(),
        &recorded.meta,
        recorded.dtype,
    ) == (
        current.handle.memory.descriptor(),
        current.handle.service,
        current.handle.offset_start,
        current.handle.offset_end,
        current.handle.size(),
        &current.meta,
        current.dtype,
    )
}

pub(in crate::simulation) fn copy(device: &Device, source: &CubeTensor, target: &CubeTensor) {
    cubecl::std::tensor::copy_into(
        &device.client,
        source.clone().binding(),
        target.clone().binding(),
        u32::elem_type_native(),
    );
}

pub(in crate::simulation) fn copy_contiguous(
    device: &Device,
    source: &CubeTensor,
    target: &CubeTensor,
) {
    match source.meta.num_elements().is_multiple_of(4) {
        true => ContiguousCopy::launch_elements(
            device,
            source.meta.num_elements().div_ceil(32),
            CopyInputLaunch {
                source: source.buffer_arg(),
                target: target.buffer_arg(),
                copies: 8,
            },
        ),
        false => copy(device, source, target),
    }
}

#[derive(Clone, Default)]
pub(in crate::simulation) struct AuditedBuffer {
    segments: Vec<CubeTensor>,
    capacity: usize,
}

struct SnapshotSegment {
    offset: usize,
    live: CubeTensor,
    audited: CubeTensor,
}

fn word_slice(tensor: &CubeTensor, offset: usize, words: usize) -> CubeTensor {
    let length = tensor.meta.num_elements();
    let handle = tensor
        .handle
        .clone()
        .offset_start((offset * size_of::<u32>()) as u64)
        .offset_end(((length - offset - words) * size_of::<u32>()) as u64);
    CubeTensor::new_contiguous(
        tensor.client.clone(),
        tensor.device.clone(),
        Shape::new([words]),
        handle,
        DType::U32,
    )
}

impl AuditedBuffer {
    pub(in crate::simulation) fn additional_words(&self, words: usize) -> usize {
        words
            .saturating_sub(self.capacity)
            .next_multiple_of(SNAPSHOT_BLOCK_WORDS)
    }

    pub(in crate::simulation) fn extended(&self, device: &Device, words: usize) -> Self {
        let mut next = self.clone();
        let additional = self.additional_words(words);
        if additional > 0 {
            next.segments.push(device.state_buffer(additional));
            next.capacity += additional;
        }
        next
    }

    fn parts(&self, live: &CubeTensor) -> impl Iterator<Item = SnapshotSegment> {
        self.segments.iter().scan(0, |offset, segment| {
            let words = segment
                .meta
                .num_elements()
                .min(live.meta.num_elements() - *offset);
            let part = SnapshotSegment {
                offset: *offset,
                live: word_slice(live, *offset, words),
                audited: word_slice(segment, 0, words),
            };
            *offset += words;
            Some(part)
        })
    }

    pub(in crate::simulation) fn read_ticks(
        &self,
        device: &Device,
        live: &CubeTensor,
    ) -> Result<Vec<u64>> {
        let mut ticks = Vec::with_capacity(live.meta.num_elements() / 2);
        for part in self.parts(live) {
            device.read_ticks_into(&part.audited, &mut ticks)?;
        }
        Ok(ticks)
    }

    pub(in crate::simulation) fn restore(&self, device: &Device, live: &CubeTensor) {
        for part in self.parts(live) {
            copy_contiguous(device, &part.audited, &part.live);
        }
    }

    #[cfg(test)]
    pub(in crate::simulation) fn shares_prefix(&self, previous: &Self) -> bool {
        self.segments.len() >= previous.segments.len()
            && self
                .segments
                .iter()
                .zip(&previous.segments)
                .all(|(a, b)| same_buffer(a, b))
    }
}

#[derive(Clone)]
pub(in crate::simulation) enum SnapshotCopy {
    Full,
    Sparse {
        nonzero: CubeTensor,
        current: CubeTensor,
    },
}

impl SnapshotCopy {
    pub(in crate::simulation) fn new(device: &Device, words: usize) -> Self {
        let properties = device.client.properties();
        match words >= 1024
            && words.is_multiple_of(4)
            && properties
                .features
                .plane
                .contains(cubecl::features::Plane::Ops)
            && properties.hardware.plane_size_min >= 32
        {
            true => Self::Sparse {
                nonzero: device.dedicated_tensor(&vec![1; words.div_ceil(SNAPSHOT_BLOCK_WORDS)]),
                current: device.dedicated_tensor(&vec![1; words.div_ceil(SNAPSHOT_BLOCK_WORDS)]),
            },
            false => Self::Full,
        }
    }

    #[cfg(test)]
    pub(in crate::simulation) fn save(
        &self,
        device: &Device,
        source: &CubeTensor,
        target: &CubeTensor,
    ) {
        self.save_range(device, source, target, 0, 0, 0);
    }

    pub(in crate::simulation) fn save_state(
        &self,
        device: &Device,
        source: &CubeTensor,
        target: &AuditedBuffer,
        stride: usize,
        ring: usize,
    ) {
        let start = crate::domain::BASE_FIELDS * stride * 2;
        let end = (crate::domain::BASE_FIELDS + 2 * ring * crate::domain::Good::COUNT) * stride * 2;
        for part in target.parts(source) {
            self.save_range(
                device,
                &part.live,
                &part.audited,
                start.saturating_sub(part.offset),
                end.saturating_sub(part.offset),
                part.offset / SNAPSHOT_BLOCK_WORDS,
            );
        }
    }

    pub(in crate::simulation) fn ring_mask(&self) -> Option<&CubeTensor> {
        match self {
            Self::Full => None,
            Self::Sparse { current, .. } => Some(current),
        }
    }

    fn save_range(
        &self,
        device: &Device,
        source: &CubeTensor,
        target: &CubeTensor,
        ring_start: usize,
        ring_end: usize,
        group_offset: usize,
    ) {
        let hardware = &device.client.properties().hardware;
        let workgroup = match (
            hardware.max_units_per_cube >= 1024,
            hardware.max_cube_dim.0 >= 1024,
        ) {
            (true, true) => 1024,
            _ => 256,
        };
        match self {
            Self::Full => copy_contiguous(device, source, target),
            Self::Sparse { nonzero, current }
                if matches!(device.cube, cubecl::Device::Metal(_))
                    && hardware.plane_size_min == 32
                    && hardware.plane_size_max == 32
                    && ring_start.div_ceil(SNAPSHOT_BLOCK_WORDS)
                        < ring_end.min(source.meta.num_elements()) / SNAPSHOT_BLOCK_WORDS =>
            {
                let first = ring_start.next_multiple_of(SNAPSHOT_BLOCK_WORDS);
                let past = ring_end.min(source.meta.num_elements()) / SNAPSHOT_BLOCK_WORDS
                    * SNAPSHOT_BLOCK_WORDS;
                let groups = (past - first) / SNAPSHOT_BLOCK_WORDS;
                for (start, end) in [(0, first), (past, source.meta.num_elements())] {
                    if start < end {
                        self.save_range(
                            device,
                            &word_slice(source, start, end - start),
                            &word_slice(target, start, end - start),
                            0,
                            0,
                            group_offset + start / SNAPSHOT_BLOCK_WORDS,
                        );
                    }
                }
                let dim = CubeDim::new_1d(256);
                copy_ring_blocks::launch(
                    &device.client,
                    cubecl::calculate_cube_count_elemwise(&device.client, groups, dim),
                    dim,
                    word_slice(source, first, past - first).buffer_arg(),
                    word_slice(target, first, past - first).buffer_arg(),
                    nonzero.buffer_arg(),
                    current.buffer_arg(),
                    (group_offset + first / SNAPSHOT_BLOCK_WORDS) as u32,
                    groups as u32,
                );
            }
            Self::Sparse { nonzero, current } => SparseCopy::launch(
                device,
                cubecl::calculate_cube_count_elemwise(
                    &device.client,
                    source.meta.num_elements().div_ceil(SNAPSHOT_PARTS * 4),
                    CubeDim::new_1d(workgroup),
                ),
                CubeDim::new_1d(workgroup),
                SnapshotInputLaunch {
                    source: source.buffer_arg(),
                    target: target.buffer_arg(),
                    nonzero: nonzero.buffer_arg(),
                    current: current.buffer_arg(),
                    groups: source.meta.num_elements().div_ceil(SNAPSHOT_PARTS * 32 * 4) as u32,
                    parts: SNAPSHOT_PARTS as u32,
                    ring_start: ring_start as u32,
                    ring_end: ring_end as u32,
                    group_offset: group_offset as u32,
                },
            ),
        }
    }
}

#[derive(CubeType, CubeLaunch)]
struct SnapshotInput {
    source: Box<[Vector<u32, Const<4>>]>,
    target: Box<[Vector<u32, Const<4>>]>,
    nonzero: Box<[u32]>,
    current: Box<[Atomic<u32>]>,
    groups: u32,
    ring_start: u32,
    ring_end: u32,
    group_offset: u32,
    #[cube(comptime)]
    parts: u32,
}

struct SparseCopy;
struct SparseWords;

impl System for SparseCopy {
    type Input = SnapshotInput;
    type Passes = gpu_passes![SparseWords];
}

#[cube]
impl GpuPass<SnapshotInput> for SparseWords {
    type Output = ();

    fn execute(mut input: SnapshotInput) {
        let group = CUBE_POS * (CUBE_DIM as usize / 32) + UNIT_POS as usize / 32;
        if group < input.groups as usize {
            let mask = input.group_offset as usize + group;
            let previous = input.nonzero[mask];
            let start = group * (comptime!(input.parts) as usize * 32 * 4);
            let ring = start >= input.ring_start as usize
                && start + comptime!(input.parts) as usize * 32 * 4 <= input.ring_end as usize;
            let observed_zero = ring && input.current[mask].load() == 0;
            if PLANE_DIM != 32 || !observed_zero || previous > 0 {
                let mut local = 0u32;
                #[unroll]
                for part in 0..comptime!(input.parts) {
                    let index = group * (comptime!(input.parts) as usize * 32)
                        + part as usize * 32
                        + UNIT_POS as usize % 32;
                    if index < input.source.len() && (!observed_zero || previous > 0) {
                        let mut value = Vector::<u32, Const<4>>::new(0);
                        if !observed_zero {
                            value = input.source[index];
                        }
                        let nonzero = value.extract(0usize)
                            | value.extract(1usize)
                            | value.extract(2usize)
                            | value.extract(3usize);
                        local |= nonzero;
                        if (nonzero | previous) != 0 {
                            input.target[index] = value;
                        }
                    }
                }
                let current = plane_any(local != 0);
                if UNIT_POS % 32 == 0 {
                    input.nonzero[mask] = u32::cast_from(current);
                }
            }
        }
    }
}

#[cube(launch)]
fn copy_ring_blocks(
    source: &[Vector<u32, Const<4>>],
    target: &mut [Vector<u32, Const<4>>],
    previous: &mut [u32],
    current: &[u32],
    offset: u32,
    groups: u32,
) {
    let group = ABSOLUTE_POS as u32;
    let lane = UNIT_POS_PLANE;
    let mut present = 0u32;
    let mut changed = false;
    if group < groups {
        let mask = (offset + group) as usize;
        present = current[mask];
        changed = (present | previous[mask]) != 0;
        previous[mask] = present;
    }
    let populated = plane_ballot(present != 0).extract(0usize);
    let mut pending = plane_ballot(changed).extract(0usize);
    let first = group - lane;
    while pending != 0 {
        let next = pending.trailing_zeros();
        let copy = (populated & (1u32 << next)) != 0;
        let base = (first + next) * comptime!(SNAPSHOT_PARTS as u32 * 32) + lane;
        #[unroll]
        for part in 0..SNAPSHOT_PARTS {
            let index = (base + part as u32 * 32) as usize;
            let mut value = Vector::<u32, Const<4>>::new(0);
            if copy {
                value = source[index];
            }
            target[index] = value;
        }
        pending &= pending - 1;
    }
}

#[derive(CubeType, CubeLaunch)]
struct CopyInput {
    source: Box<[Vector<u32, Const<4>>]>,
    target: Box<[Vector<u32, Const<4>>]>,
    #[cube(comptime)]
    copies: u32,
}

struct ContiguousCopy;
struct CopyWords;

impl System for ContiguousCopy {
    type Input = CopyInput;
    type Passes = gpu_passes![CopyWords];
}

#[cube]
impl GpuPass<CopyInput> for CopyWords {
    type Output = ();

    fn execute(mut input: CopyInput) {
        let first = ABSOLUTE_POS as u32 / CUBE_DIM * CUBE_DIM * comptime!(input.copies) + UNIT_POS;
        #[unroll]
        for part in 0..comptime!(input.copies) {
            let index = (first + part * CUBE_DIM) as usize;
            if index < input.source.len() {
                input.target[index] = input.source[index];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_gpu_snapshot_preserves_transitions_and_restore() -> Result<()> {
        let small = crate::simulation::tests::device(32)?;
        let large = crate::simulation::tests::device(256)?;
        for (count, device) in [
            (1, &small),
            (2, &small),
            (3, &small),
            (4, &small),
            (255, &small),
            (260, &small),
            (1020, &small),
            (1024, &small),
            (1028, &small),
            (2048, &small),
            (2052, &small),
            (4092, &large),
            (4096, &large),
            (4100, &large),
            (65_540, &large),
            (65_541, &large),
        ] {
            let snapshot = SnapshotCopy::new(device, count);
            let audited = device.tensor(&vec![u32::MAX; count]);
            let initial = vec![0; count];
            let populated = (0..count)
                .map(|index| match index / 1024 % 3 {
                    0 => (index as u32).wrapping_mul(2_654_435_761),
                    _ => 0,
                })
                .collect::<Vec<_>>();
            let changed = (0..count)
                .map(|index| match index / 1024 % 3 {
                    1 => u32::MAX - index as u32,
                    _ => 0,
                })
                .collect::<Vec<_>>();
            for values in [&initial, &populated, &changed, &initial, &populated] {
                let source = device.tensor(values);
                snapshot.save(device, &source, &audited);
                assert_eq!(device.read(&audited)?, *values);
                let restored = device.tensor(&vec![17; count]);
                copy_contiguous(device, &audited, &restored);
                assert_eq!(device.read(&restored)?, *values);
            }
        }
        Ok(())
    }

    #[test]
    fn audited_zero_ring_blocks_preserve_snapshot_transitions_and_boundaries_on_gpu() -> Result<()>
    {
        let device = crate::simulation::tests::device(64)?;
        for count in [1024, 2052, 65_540] {
            let snapshot = SnapshotCopy::new(&device, count);
            let audited = device.tensor(&vec![u32::MAX; count]);
            let mask = snapshot.ring_mask().expect("sparse snapshot mask");
            let start = SNAPSHOT_BLOCK_WORDS / 2;
            let end = count - SNAPSHOT_BLOCK_WORDS / 2;
            for populated in [0usize, 1, 2, 0, 1] {
                let values = (0..count)
                    .map(
                        |index| match index / SNAPSHOT_BLOCK_WORDS % 3 == populated {
                            true => index as u32 + 1,
                            false => 0,
                        },
                    )
                    .collect::<Vec<_>>();
                let source = device.tensor(&values);
                let observed = values
                    .chunks(SNAPSHOT_BLOCK_WORDS)
                    .map(|block| u32::from(block.iter().any(|&value| value > 0)))
                    .collect::<Vec<_>>();
                device.write(mask, &observed);
                snapshot.save_range(&device, &source, &audited, start, end, 0);
                assert_eq!(device.read(&audited)?, values);
                reset_ring_mask(&device, mask, RingMaskState::Unknown);
                snapshot.save_range(&device, &source, &audited, start, end, 0);
                assert_eq!(device.read(&audited)?, values);
            }
        }
        Ok(())
    }

    #[test]
    fn sparse_gpu_snapshot_ignores_dispatch_groups_past_the_logical_extent() -> Result<()> {
        let device = crate::simulation::tests::device(64)?;
        let group_words = SNAPSHOT_PARTS * 32 * 4;
        let source = device.tensor(&vec![17; group_words * 2]);
        let sentinel = 0xdead_beef;
        let target = device.tensor(&vec![sentinel; group_words * 2]);
        let metadata = device.tensor(&[1, sentinel]);
        let current = device.tensor(&[1]);
        SparseCopy::launch(
            &device,
            CubeCount::Static(2, 1, 1),
            CubeDim::new_1d(32),
            SnapshotInputLaunch {
                source: source.into_buffer_arg(),
                target: target.buffer_arg(),
                nonzero: metadata.buffer_arg(),
                current: current.into_buffer_arg(),
                groups: 1,
                parts: SNAPSHOT_PARTS as u32,
                ring_start: 0,
                ring_end: 0,
                group_offset: 0,
            },
        );
        assert_eq!(
            device.read_range(&target, 0, group_words)?,
            vec![17; group_words],
        );
        assert_eq!(
            device.read_range(&target, group_words, group_words)?,
            vec![sentinel; group_words],
        );
        assert_eq!(device.read(&metadata)?, [1, sentinel]);
        Ok(())
    }
}
